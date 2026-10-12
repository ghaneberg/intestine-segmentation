/*
 * export_labeled_masks.groovy
 * ------------------------------------------------------------------
 * Export one integer-labeled mask TIFF per slide (a per-pixel label
 * image) from the class annotations created by the pixel classifier.
 * These masks feed BOTH the 3D reconstruction and nnU-Net.
 *
 * Label mapping (matches AIM doc / nnU-Net config):
 *   0 = background (outside LeftCurl / unclassified)
 *   1 = Lumen   2 = Mucosa   3 = Submucosa   4 = Muscularis   5 = Serosa
 *
 * HOW TO RUN:
 *   Automate -> Script editor -> File -> Open this file
 *   Run -> "Run for project"   (writes all slides; skips ones w/o classes)
 *
 * Output: <project>/labeled_masks/<slidename>_labels.tif
 * ------------------------------------------------------------------
 */

import qupath.lib.images.servers.LabeledImageServer

// ===== CONFIG ======================================================
double DOWNSAMPLE = 1.0   // 1.0 = full resolution (masks align to the TIFFs).
                          // Raise to 2/4 for smaller files if needed.
// ===================================================================

def imageData = getCurrentImageData()
def entry     = getProjectEntry()
def name      = entry != null ? entry.getImageName()
                              : imageData.getServer().getMetadata().getName()

// Skip slides that have no class annotations (e.g. the 8 missing indices)
def classNames = ['Lumen', 'Mucosa', 'Submucosa', 'Muscularis', 'Serosa'] as Set
def hasClasses = getAnnotationObjects().any {
    def pc = it.getPathClass(); pc != null && classNames.contains(pc.getName())
}
if (!hasClasses) {
    println "SKIP (no class annotations): ${name}"
    return
}

def outputDir = buildFilePath(PROJECT_BASE_DIR, 'labeled_masks')
mkdirs(outputDir)

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE)   // everything unlabeled -> 0
    .downsample(DOWNSAMPLE)
    .addLabel('Lumen',      1)
    .addLabel('Mucosa',     2)
    .addLabel('Submucosa',  3)
    .addLabel('Muscularis', 4)
    .addLabel('Serosa',     5)
    .multichannelOutput(false)              // single-channel indexed labels
    .build()

def base = name.replaceAll(/\.t[if]+f?$/, '')   // strip .tif/.tiff
def outputPath = buildFilePath(outputDir, base + '_labels.tif')
writeImage(labelServer, outputPath)
println "exported: ${base}_labels.tif"
