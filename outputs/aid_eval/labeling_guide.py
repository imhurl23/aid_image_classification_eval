"""Operational web-image labeling rubric; not an official AID annotation manual."""

GUIDE = """## Label the image once
Choose the **overall scene**, not every object present. Your category will be copied to all five model results automatically. Do not use a model answer or the search term as ground truth.

1. Decide whether this is a usable aerial photograph. Oblique aerial views are allowed.
2. Identify the dominant scene or coherent facility. A small incidental feature does not define the whole image.
3. Choose a category only when the image provides enough evidence. You do not need to force a guess.
4. If two categories are equally defensible, choose **Ambiguous**. You can correct a label later; the correction will propagate too.

### When to abstain
- **Ambiguous:** recognizable aerial image, but multiple classes fit equally well, or the facility/density cannot be identified confidently.
- **Other:** usable aerial scene clearly outside the 30 classes, such as an isolated golf course or open ocean.
- **NotAerial:** ground-level or interior photograph, not an overhead or elevated aerial view.
- **Unusable:** map, diagram, collage, unreadable image, severe obstruction, or too little visual evidence to judge.

### Common confusing pairs
- **Commercial / Industrial:** shops, offices and customer-facing premises versus factories, production yards and processing infrastructure. A large roof alone is insufficient; use Ambiguous when purpose is unclear.
- **School / Residential:** a recognizable campus with coordinated buildings and grounds versus housing. A sports field beside buildings does not establish a school.
- **Bridge / Viaduct:** a localized crossing versus an extended elevated road/rail structure, often with repeated spans or interchange ramps. If structure/extent is unclear, use Ambiguous.
- **Resort / Port / Beach:** an integrated leisure/hotel complex versus a harbor dominated by docks/boats/cargo versus a shoreline dominated by beach. Incidental pools, boats or sand do not override the overall scene.
- **Center / Square / Commercial:** a broad civic/urban-center complex versus a dominant open paved plaza versus commercial premises. These overlap: use the AID examples below and choose Ambiguous when the distinction is not visually supported.
- **Residential density:** compare visible spacing between homes and open land, not building height. No numerical density thresholds are asserted here; borderline cases are Ambiguous.

### Category reference
| Category | Visual evidence to look for |
|---|---|
| Airport | Runways/taxiways, aprons and airport facilities |
| BareLand | Exposed soil/rock with little vegetation or development |
| BaseballField | Recognizable baseball diamond and outfield |
| Beach | Sand/shingle shoreline is the principal scene |
| Bridge | A distinct road/rail crossing over water or another obstacle |
| Center | Urban/civic-center complex; compare official examples, abstain if unclear |
| Church | Visually identifiable church complex; do not infer religion from generic roofs |
| Commercial | Retail/office/business premises supported by surrounding layout |
| DenseResidential | Tightly packed housing, little separation/open space |
| Desert | Extensive arid landscape, dunes or desert terrain |
| Farmland | Cultivated plots, crop rows or agricultural field patterns |
| Forest | Predominantly continuous tree canopy |
| Industrial | Manufacturing/processing facilities and associated yards |
| Meadow | Open grassland without clear crop-field structure |
| MediumResidential | Housing with moderate spacing, gardens and local roads |
| Mountain | Dominant rugged/high-relief landforms |
| Park | Managed recreational green space with paths/landscaping |
| Parking | Vehicle parking area dominates |
| Playground | Outdoor play/sports grounds; distinguish stadium seating and baseball diamonds |
| Pond | Small enclosed standing-water body dominates |
| Port | Harbor/docks/berths and maritime facilities dominate |
| RailwayStation | Rail platforms/station complex and track approaches |
| Resort | Coherent leisure/accommodation complex with recreational amenities |
| River | Elongated natural watercourse dominates |
| School | Recognizable educational campus with supporting grounds/layout |
| SparseResidential | Widely separated homes with substantial open land |
| Square | Open paved civic plaza dominates |
| Stadium | Sports arena/field with substantial spectator stands |
| StorageTanks | Recognizable grouped large storage tanks dominate |
| Viaduct | Extended elevated road/rail structure or repeated spans |

This is our operational rubric for **web images**, not a claim that every web photo has a unique AID label. The original AID examples were selected and labeled by remote-sensing specialists. [Official AID class examples](https://captain-whu.github.io/AID/) are useful visual references; the distinctions above are practical review guidance, not official quantitative thresholds.
"""
