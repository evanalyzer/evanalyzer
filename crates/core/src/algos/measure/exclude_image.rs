//! # Exclude image
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-10-22
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **AGPL-3.0**.

use crate::algos::{ExecutionScope, GlobalPipelineCache, ImageAlgorithm, PipelineContext};
use crate::pipeline::pipeline_measure::{MeasurementRegion, MeasurementValues};
use evanalyzer_cfg::core_types::{CitationMetadata, InternalErrors};
use macros::CommandsMeta;

/// Leaves an image out of the results when its measurements say it's bad.
///
/// Each rule compares a measurement of an earlier step (e.g. Image
/// Quality's focus score) with a value. When the rules are met, the image
/// is analysed to the end but disabled in the results, with the reason - to
/// check it by hand and enable it again if it's fine. The rules are checked
/// on every plane; one plane meeting them excludes the whole image. A
/// measurement without a value (e.g. the slope of a flat image) doesn't
/// meet a rule.
#[derive(CommandsMeta)]
#[cmdsmeta(category = "measure", display_name = "Exclude Image")]
pub struct ExcludeImage {
    /// Excluded when one rule is met, or only when all of them are.
    #[cmdsmeta(summary = true)]
    pub exclude_when: ExcludeWhen,

    /// The rules: a measurement below or above a value.
    #[cmdsmeta(display_name = "Rules")]
    pub rules: Vec<ExcludeRule>,
}

/// How the rules combine.
#[derive(CommandsMeta)]
pub enum ExcludeWhen {
    /// One rule met is enough.
    AnyRule,
    /// Every rule must be met.
    AllRules,
}

/// One rule: a measurement compared with a value.
#[derive(CommandsMeta)]
pub struct ExcludeRule {
    /// The measurement, made by an earlier step.
    #[cmdsmeta(measurement, summary = true)]
    pub measurement: String,

    /// Met when the measurement is below or above the value.
    #[cmdsmeta(summary = true)]
    pub comparison: RuleComparison,

    /// The value, in the measurement's unit.
    #[cmdsmeta(summary = true)]
    pub value: f64,
}

/// Which side of the value meets a rule.
#[derive(CommandsMeta)]
pub enum RuleComparison {
    /// Met below the value (e.g. a focus score too low).
    Below,
    /// Met above the value (e.g. too many saturated pixels).
    Above,
}

impl ImageAlgorithm for ExcludeImage {
    fn execute(
        &self,
        _ctx: &mut PipelineContext,
        cache: &mut GlobalPipelineCache,
    ) -> Result<(), InternalErrors> {
        if self.rules.is_empty() {
            return Ok(());
        }
        let mut met = Vec::new();
        let mut reasons = Vec::new();
        for (i, rule) in self.rules.iter().enumerate() {
            if rule.measurement.is_empty() {
                return Err(InternalErrors::InvalidArgument(format!(
                    "Exclude image: rule {} has no measurement - choose one",
                    i + 1
                )));
            }
            let measurement = cache
                .measurements
                .measurements
                .iter()
                .find(|m| m.info.key == rule.measurement)
                .ok_or_else(|| {
                    InternalErrors::InvalidArgument(format!(
                        "Exclude image: \"{}\" isn't measured before this step - put the step \
                         that measures it first",
                        rule.measurement
                    ))
                })?;
            let value = match &measurement.values {
                MeasurementValues::Image(values) => values
                    .iter()
                    .find(|(region, _)| *region == MeasurementRegion::Image)
                    .map(|(_, v)| *v),
                MeasurementValues::Objects(_) => {
                    return Err(InternalErrors::InvalidArgument(format!(
                        "Exclude image: \"{}\" has a value per object - a rule needs one per \
                         image",
                        measurement.info.name
                    )));
                }
            };
            let is_met = value.is_some_and(|v| match rule.comparison {
                RuleComparison::Below => v < rule.value,
                RuleComparison::Above => v > rule.value,
            });
            if let (true, Some(v)) = (is_met, value) {
                let side = match rule.comparison {
                    RuleComparison::Below => "below",
                    RuleComparison::Above => "above",
                };
                reasons.push(format!(
                    "{} of {} = {}, {side} {}",
                    measurement.info.name,
                    measurement.info.step,
                    format_number(v),
                    format_number(rule.value)
                ));
            }
            met.push(is_met);
        }
        let excluded = match self.exclude_when {
            ExcludeWhen::AnyRule => met.iter().any(|m| *m),
            ExcludeWhen::AllRules => met.iter().all(|m| *m),
        };
        if excluded {
            let reason = reasons.join("; ");
            // A second Exclude Image step adds its reason.
            cache.measurements.excluded = Some(match cache.measurements.excluded.take() {
                Some(earlier) => format!("{earlier}; {reason}"),
                None => reason,
            });
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Exclude image"
    }

    fn cite(&self) -> Vec<&'static CitationMetadata> {
        vec![&CitationMetadata::DANMAYR]
    }

    /// After the measurement steps, on the whole image.
    fn execution_scope(&self) -> ExecutionScope {
        ExecutionScope::WholeImage
    }

    /// Runs on every plane, also one without objects (like the measurement
    /// steps it reads).
    fn measures(&self) -> bool {
        true
    }
}

/// A value as the reason shows it: 4 significant digits.
fn format_number(v: f64) -> String {
    if v != 0.0 && (v.abs() >= 1e5 || v.abs() < 1e-3) {
        format!("{v:.3e}")
    } else {
        let digits = (3 - v.abs().log10().floor().max(-3.0) as i32).clamp(0, 6) as usize;
        let text = format!("{v:.digits$}");
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algos::{ImageQuality, MeasureColocalization};
    use crate::pipeline::pipeline::PipelineImageMeta;
    use evanalyzer_cfg::core_types::{ImageChannelIdx, ObjectId};

    /// A plane where quality_01 measured channel 0: focus score 0.09, 3 %
    /// saturated, no slope (a flat image).
    fn cache() -> GlobalPipelineCache {
        let mut cache = GlobalPipelineCache::default();
        let quality = ImageQuality {
            id: "quality_01".into(),
            channel: ImageChannelIdx(0),
        };
        let values = [
            ("focus_score", Some(0.09)),
            ("power_log_log_slope", None),
            ("percent_maximal", Some(3.0)),
            ("percent_minimal", Some(0.0)),
            ("noise", Some(12.5)),
        ];
        for (id, value) in values {
            cache.add_image_measurement(
                quality.measurement_output(id),
                value.map(|v| (MeasurementRegion::Image, v)),
            );
        }
        cache
    }

    fn rule(key: &str, comparison: RuleComparison, value: f64) -> ExcludeRule {
        ExcludeRule {
            measurement: format!("quality_01/{key}"),
            comparison,
            value,
        }
    }

    fn run(step: &ExcludeImage, cache: &mut GlobalPipelineCache) -> Result<(), InternalErrors> {
        let mut ctx =
            PipelineContext::without_image(std::path::PathBuf::new(), PipelineImageMeta::default())
                .unwrap();
        step.execute(&mut ctx, cache)
    }

    #[test]
    fn a_rule_met_excludes_with_the_reason() {
        let mut cache = cache();
        let step = ExcludeImage {
            exclude_when: ExcludeWhen::AnyRule,
            rules: vec![
                rule("focus_score", RuleComparison::Below, 0.15),
                rule("percent_maximal", RuleComparison::Above, 5.0),
            ],
        };
        run(&step, &mut cache).unwrap();
        assert_eq!(
            cache.measurements.excluded.as_deref(),
            Some("Focus score (channel 0) of quality_01 = 0.09, below 0.15")
        );
    }

    #[test]
    fn all_rules_must_be_met_when_asked() {
        let both = |limit: f64| ExcludeImage {
            exclude_when: ExcludeWhen::AllRules,
            rules: vec![
                rule("focus_score", RuleComparison::Below, 0.15),
                rule("percent_maximal", RuleComparison::Above, limit),
            ],
        };
        let mut cache = cache();
        run(&both(5.0), &mut cache).unwrap();
        assert_eq!(cache.measurements.excluded, None, "only one met");

        run(&both(2.0), &mut cache).unwrap();
        assert_eq!(
            cache.measurements.excluded.as_deref(),
            Some(
                "Focus score (channel 0) of quality_01 = 0.09, below 0.15; \
                 Pixels at the maximum (channel 0) of quality_01 = 3, above 2"
            )
        );
    }

    #[test]
    fn a_measurement_without_a_value_meets_no_rule() {
        let mut cache = cache();
        let step = ExcludeImage {
            exclude_when: ExcludeWhen::AnyRule,
            rules: vec![rule("power_log_log_slope", RuleComparison::Below, -1.0)],
        };
        run(&step, &mut cache).unwrap();
        assert_eq!(cache.measurements.excluded, None);
    }

    #[test]
    fn no_rules_exclude_nothing() {
        let mut cache = cache();
        let step = ExcludeImage {
            exclude_when: ExcludeWhen::AllRules,
            rules: vec![],
        };
        run(&step, &mut cache).unwrap();
        assert_eq!(cache.measurements.excluded, None);
    }

    #[test]
    fn a_measurement_not_made_is_an_error() {
        let mut cache = cache();
        let missing = ExcludeImage {
            exclude_when: ExcludeWhen::AnyRule,
            rules: vec![ExcludeRule {
                measurement: "quality_02/focus_score".into(),
                comparison: RuleComparison::Below,
                value: 0.1,
            }],
        };
        let err = run(&missing, &mut cache).unwrap_err().to_string();
        assert!(err.contains("isn't measured before this step"), "{err}");

        let empty = ExcludeImage {
            exclude_when: ExcludeWhen::AnyRule,
            rules: vec![ExcludeRule {
                measurement: String::new(),
                comparison: RuleComparison::Below,
                value: 0.1,
            }],
        };
        let err = run(&empty, &mut cache).unwrap_err().to_string();
        assert!(err.contains("rule 1 has no measurement"), "{err}");

        // A value per object can't decide about an image.
        let info = MeasureColocalization {
            id: "per_object".into(),
            channel_a: ImageChannelIdx(0),
            channel_b: ImageChannelIdx(1),
            threshold_percent: 15.0,
            costes: crate::algos::CostesMethod::Off,
            objects: evanalyzer_cfg::core_types::ObjectClass::Global(1),
        }
        .measurement_output("object_pearson");
        cache.add_object_measurement(info.clone(), vec![(ObjectId(1), 0.5)]);
        let per_object = ExcludeImage {
            exclude_when: ExcludeWhen::AnyRule,
            rules: vec![ExcludeRule {
                measurement: info.key,
                comparison: RuleComparison::Below,
                value: 0.1,
            }],
        };
        let err = run(&per_object, &mut cache).unwrap_err().to_string();
        assert!(err.contains("a value per object"), "{err}");
    }

    #[test]
    fn numbers_read_well() {
        assert_eq!(format_number(0.09), "0.09");
        assert_eq!(format_number(0.123456), "0.1235");
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(1234.5678), "1235");
        assert_eq!(format_number(-1.72), "-1.72");
        assert_eq!(format_number(0.00002), "2.000e-5");
        assert_eq!(format_number(0.0), "0");
    }
}
