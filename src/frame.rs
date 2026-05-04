use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::orb_vocabulary::OrbVocabulary;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub struct Frame {
    // Current Frame id
    pub id: usize,

    // Vocabulary used for relocalization
    pub vocabulary: Arc<OrbVocabulary>,
    // Feature extractor. The right is used only in the stereo case.
    //pub extractor_left: Arc<ORB>
    // TODO: HERE
}

impl Frame {
    /*fn from_frame(frame: &Frame) -> Self {
        Frame { id: frame.id }
    }*/
}
