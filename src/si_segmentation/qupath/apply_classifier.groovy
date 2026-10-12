/*
 * apply_classifier.groovy
 * ------------------------------------------------------------------
 * Apply the trained pixel classifier to EVERY slide in the project,
 * creating per-class annotations (Lumen / Mucosa / Submucosa /
 * Muscularis / Serosa) across the WHOLE image.
 *
 * No LeftCurl gating -- unlike the old batch_apply_classifier.groovy,
 * these registered crops already have the right loop masked to pure
 * white, so the left loop is the only tissue in frame. The white
 * background classifies to nothing (or an Ignore class) and drops out
 * at the mask-export step.
 *
 * HOW TO RUN:
 *   1. EDIT CLASSIFIER_NAME below to your saved classifier's exact name.
 *   2. Test first: open one slide, Run (plain "Run"), eyeball the result.
 *   3. Then Run -> "Run for project", tick all images, let it save.
 *
 * Re-runnable: PURGE_OLD wipes previous class annotations first so a
 * re-run does not stack duplicates.
 * ------------------------------------------------------------------
 */

// ===== CONFIG (edit me) ============================================
def   CLASSIFIER_NAME = 'new_classified'   // EXACT saved pixel-classifier name
double MIN_AREA       = 3000.0         // px^2 min object size
double MIN_HOLE       = 3000.0         // px^2 min hole size
boolean PURGE_OLD     = true           // delete prior class annotations first
// ===================================================================

def CLASS_NAMES = ['Lumen', 'Mucosa', 'Submucosa', 'Muscularis', 'Serosa'] as Set

def entry = getProjectEntry()
def name  = entry != null ? entry.getImageName()
                          : getCurrentImageData().getServer().getMetadata().getName()

// Fail clearly if the classifier name is wrong, and list what IS available.
def available = getProject()?.getPixelClassifiers()?.getNames()
if (available != null && !available.contains(CLASSIFIER_NAME)) {
    println "ABORT: no classifier named '${CLASSIFIER_NAME}'. Available: ${available}"
    return
}

// 1) Optional cleanup so re-runs do not accumulate duplicate annotations
if (PURGE_OLD) {
    def old = getAnnotationObjects().findAll {
        def pc = it.getPathClass(); pc != null && CLASS_NAMES.contains(pc.getName())
    }
    if (old) removeObjects(old, true)
}

// 2) Run the classifier over the whole image
//    (no selection => full image region)
createAnnotationsFromPixelClassifier(CLASSIFIER_NAME, MIN_AREA, MIN_HOLE)

// 3) Refresh + save
fireHierarchyUpdate()
if (entry != null) entry.saveImageData(getCurrentImageData())

def counts = [:]
getAnnotationObjects().each {
    def pc = it.getPathClass()
    if (pc != null && CLASS_NAMES.contains(pc.getName()))
        counts[pc.getName()] = (counts[pc.getName()] ?: 0) + 1
}
println "done: ${name}  ${counts}"
