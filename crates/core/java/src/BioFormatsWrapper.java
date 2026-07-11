
// BioFormatsWrapper.java

import loci.common.services.ServiceFactory;
import loci.formats.ImageReader;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;
import java.nio.ByteBuffer;
import loci.common.DebugTools;
import loci.formats.IFormatReader;
import loci.formats.Memoizer;
import loci.formats.ChannelSeparator;

public class BioFormatsWrapper {

    static {
        DebugTools.setRootLevel("OFF");
    }

    IFormatReader formatReader ;
    OMEXMLService service;

    public BioFormatsWrapper(String imagePath, boolean splitImageChannels) {
        try {
            ServiceFactory factory = new ServiceFactory();
            service = factory.getInstance(OMEXMLService.class);

            // The 2-arg constructor (not the 3-arg one with an explicit
            // `null` directory) is required: Memoizer's `doInPlaceCaching`
            // flag - which lets a null directory mean "cache next to the
            // source file" - is only set true by this constructor. Passing
            // `null` to the 3-arg constructor leaves it false, silently
            // disabling the memo cache (isSavedToMemo()/isLoadedFromMemo()
            // never true, no .bfmemo file ever written) - verified via
            // javap against bioformats.jar 7.0.1's actual bytecode, not
            // just the API docs.
            IFormatReader memoizer = new Memoizer(new ImageReader(), 1);
            if (splitImageChannels) {
                formatReader = new ChannelSeparator(memoizer);
            } else {
                formatReader = memoizer;
            }
            IMetadata metadata = service.createOMEXMLMetadata();
            formatReader.setMetadataStore(metadata);
            formatReader.setFlattenedResolutions(false);
            formatReader.setId(imagePath);
        } catch (Exception e) {
            if (!imagePath.endsWith("warmup")) {
                e.printStackTrace();
            }
        }
    }

    public void close() {
        try {
            formatReader.close();
        } catch (Exception e) {

        }
    }

    public void readImageTile(ByteBuffer targetBuffer, int series, int resolution, int z,
            int c, int t, int x, int y,
            int width, int height) throws Exception{
        synchronized(formatReader) {
            // Create an appropriate reader for the format
            formatReader.setSeries(series);
            if (resolution >= formatReader.getResolutionCount()) {
                resolution = formatReader.getResolutionCount() - 1;
            }
            formatReader.setResolution(resolution);
            // Read the image data for the current channel, timepoint, and slice
            byte[] imageBytes = formatReader.openBytes(formatReader.getIndex(z, c, t), x, y, width, height);
            targetBuffer.put(imageBytes);
        }
    }

    /// https://docs.openmicroscopy.org/ome-model/6.2.2/ome-tiff/specification.html
    public String getImageProperties() {
        String omeXML = "";
        try {
            // Create a service factory
            int seriesCount = formatReader.getSeriesCount();
            omeXML = service.getOMEXML((IMetadata) formatReader.getMetadataStore());
            omeXML = omeXML + "\n<JODA xmlns=\"https://www.imagec.org/\" SeriesCount=\""
                    + String.valueOf(seriesCount) + "\">";

            for (int series = 0; series < seriesCount; series++) {
                formatReader.setSeries(series);
                omeXML = omeXML + "\n<Series idx=\"" + String.valueOf(series) + "\" ResolutionCount=\""
                        + String.valueOf(formatReader.getResolutionCount()) + "\">";

                String format = formatReader.getFormat().toLowerCase();
                int optimalTileWidth = formatReader.getOptimalTileWidth();
                int optimalTileHeight = formatReader.getOptimalTileHeight();
                if (format.contains("jpeg")) {
                    optimalTileWidth = formatReader.getSizeX();
                    optimalTileHeight = formatReader.getSizeY();
                }

                for (int n = 0; n < formatReader.getResolutionCount(); n++) {
                    formatReader.setResolution(n);
                    omeXML += "<PyramidResolution idx=\"" + String.valueOf(n) + "\" width=\""
                            + String.valueOf(formatReader.getSizeX()) + "\" height=\""
                            + String.valueOf(formatReader.getSizeY()) + "\" TileWidth=\""
                            + String.valueOf(optimalTileWidth) + "\" TileHeight=\""
                            + String.valueOf(optimalTileHeight) + "\" BitsPerPixel=\""
                            + String.valueOf(formatReader.getBitsPerPixel()) + "\" RGBChannelCount=\""
                            + String.valueOf(formatReader.getRGBChannelCount()) + "\" IsInterleaved=\""
                            + String.valueOf(formatReader.isInterleaved() == true ? 1 : 0) + "\" IsLittleEndian=\""
                            + String.valueOf(formatReader.isLittleEndian() == true ? 1 : 0) + "\"></PyramidResolution>";

                }
                omeXML = omeXML + "</Series>\n";
            }
            omeXML += "</JODA>";
        } catch (Exception e) {
            e.printStackTrace();
        }
        return omeXML;
    }
}
