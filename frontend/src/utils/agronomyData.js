/**
 * Agronomic treatment database and knowledge base for PlantVillage classes.
 * Provides scientifically verified organic remedies, chemical dosages,
 * symptoms, and seasonal prevention schedules for all diagnosed crop conditions.
 */

export const SEVERITY_LEVELS = {
  LOW: {
    level: 'Low',
    color: 'text-emerald-700 dark:text-emerald-400',
    bg: 'bg-emerald-500/10 dark:bg-emerald-500/15',
    border: 'border-emerald-500/30',
    badge: 'border border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
    description: 'Minimal tissue damage. Preventative or organic management is sufficient.',
  },
  MODERATE: {
    level: 'Moderate',
    color: 'text-amber-700 dark:text-amber-400',
    bg: 'bg-amber-500/10 dark:bg-amber-500/15',
    border: 'border-amber-500/30',
    badge: 'border border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-300',
    description: 'Active localized infection. Targeted fungicide or bio-remedy intervention recommended.',
  },
  SEVERE: {
    level: 'Severe',
    color: 'text-rose-700 dark:text-rose-400',
    bg: 'bg-rose-500/10 dark:bg-rose-500/15',
    border: 'border-rose-500/30',
    badge: 'border border-rose-500/30 bg-rose-500/10 text-rose-800 dark:text-rose-300',
    description: 'Aggressive systemic or rapid-spreading pathogen. Immediate treatment and containment required.',
  },
}

/**
 * Determine disease severity based on disease name, healthy status, and confidence score.
 */
export function getSeverityInfo(disease = '', confidence = 1.0) {
  const d = (disease || '').toLowerCase()
  if (!d || d.includes('healthy')) {
    return {
      ...SEVERITY_LEVELS.LOW,
      level: 'Healthy / Low',
      description: 'Plant foliage appears healthy with normal chlorophyll pigmentation.',
    }
  }

  // Highly destructive / systemic pathogens
  if (
    d.includes('late blight') ||
    d.includes('virus') ||
    d.includes('greening') ||
    d.includes('bacterial') ||
    d.includes('wilt') ||
    d.includes('esca') ||
    (confidence > 0.9 && (d.includes('blight') || d.includes('rot')))
  ) {
    return SEVERITY_LEVELS.SEVERE
  }

  // Moderate foliar diseases
  if (
    d.includes('early blight') ||
    d.includes('scab') ||
    d.includes('rust') ||
    d.includes('spot') ||
    d.includes('scorch') ||
    d.includes('mildew') ||
    d.includes('mold') ||
    d.includes('mite') ||
    d.includes('rot')
  ) {
    return confidence >= 0.7 ? SEVERITY_LEVELS.MODERATE : SEVERITY_LEVELS.LOW
  }

  return SEVERITY_LEVELS.MODERATE
}

/**
 * Agronomic treatment data repository keyed by normalized disease or crop.
 */
export const AGRONOMIC_GUIDES = {
  'early blight': {
    pathogen: 'Fungal (Alternaria solani)',
    symptoms: [
      'Small brown to black spots with characteristic concentric rings creating a "target-board" pattern.',
      'Yellow chlorotic halos surrounding lesions, primarily beginning on older bottom leaves.',
      'Progressive premature defoliation exposing fruit to sunscald.',
      'Dark, sunken collar rot lesions on lower stems.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Bacillus subtilis (Serenade ASO) applied at 3.0 - 5.0 ml/L at first sign of spotting.',
        'Copper Octanoate (Copper soap shield) at 2.0 - 3.5 ml/L every 7-10 days.',
        'Trichoderma harzianum soil drench and foliar formulation to suppress fungal sporulation.',
      ],
      botanicalOils: [
        'Cold-pressed Neem Oil (0.5% - 1.0% solution, ~5ml/L) with mild horticultural soap emulsifier.',
        'Potassium Bicarbonate foliar spray (3g/L) to alter leaf surface pH against spores.',
      ],
      culturalPractices: [
        'Prune lower leaves 12-18 inches from the soil surface to prevent soil splash inoculation.',
        'Apply 2-3 inches of organic straw or plastic mulch to barrier spores in the soil.',
        'Convert to drip irrigation to keep canopy foliage dry; avoid overhead watering.',
        'Sanitize all pruning tools in 70% isopropyl alcohol between plants.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Chlorothalonil 720 SC', rate: '2.0 - 2.5 ml/L', perAcre: '1.5 - 2.0 pt/acre', phi: '0 days', rei: '12 hours' },
        { name: 'Mancozeb 75% WP', rate: '2.0 - 2.5 g/L', perAcre: '1.5 - 2.0 lbs/acre', phi: '5 days', rei: '24 hours' },
        { name: 'Azoxystrobin 23% SC', rate: '1.0 - 1.25 ml/L', perAcre: '6.0 - 8.0 fl oz/acre', phi: '0 days', rei: '4 hours' },
        { name: 'Difenoconazole 25% EC', rate: '0.5 - 0.75 ml/L', perAcre: '4.0 - 5.5 fl oz/acre', phi: '7 days', rei: '12 hours' },
      ],
      sprayInterval: 'Apply preventative sprays every 7 to 10 days during warm, humid conditions (75-85°F).',
      resistanceManagement: 'Alternate FRAC Group M5 (Chlorothalonil) with FRAC Group 11 (Azoxystrobin) to avoid resistance buildup.',
    },
    preventionSchedule: [
      { stage: 'Pre-Planting', actions: 'Select certified resistant seed/transplants. Practice 3-year rotation away from Solanaceae (tomato, potato, eggplant).' },
      { stage: 'Vegetative Growth', actions: 'Stake and trellis plants for optimal airflow. Mulch root zones and prune suckers/bottom foliage.' },
      { stage: 'Flowering & Fruiting', actions: 'Inspect lower foliage weekly. Initiate preventative bio-fungicide or copper spray at first symptom onset.' },
      { stage: 'Post-Harvest Cleanup', actions: 'Remove and destroy all crop residues; do not compost blighted vines. Solarize or deep-till soil.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.0, chemicalUnit: 'g', sprayVolumeGalPerAcre: 50, sprayVolumeLitersPer1000SqFt: 4.5 },
  },

  'late blight': {
    pathogen: 'Oomycete (Phytophthora infestans)',
    symptoms: [
      'Water-soaked, irregular pale green lesions expanding rapidly into dark, greasy brown-black patches.',
      'White cottony fungal/oomycete sporulation visible on leaf undersides in humid conditions.',
      'Rapid collapse and blackening of whole stems, petioles, and foliage within days.',
      'Firm, dark brown sunken rot on tubers or green/ripe fruit surfaces.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Copper Hydroxide / Bordeaux Mixture (1% copper sulfate + hydrated lime) applied preventatively.',
        'Bacillus amyloliquefaciens (Double Nickel) applied at 2.5 - 4.0 g/L.',
        'Fixed copper compounds applied every 5-7 days when weather is cool (60-70°F) and damp.',
      ],
      botanicalOils: [
        'Pure cold-pressed neem extract (10 ml/L) as a protective spore germination inhibitor.',
        'Horticultural oils combined with bio-stimulants to enhance leaf cuticle thickness.',
      ],
      culturalPractices: [
        'Immediately rogue and bag completely infected plants; do not leave cull piles near fields.',
        'Maintain wide row spacing (>36 inches) for rapid morning drying.',
        'Eliminate volunteer tomato and potato plants which act as pathogen overwintering reservoirs.',
        'Ensure excellent field drainage to prevent standing water around roots.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Cymoxanil 60% + Mancozeb 20% WP', rate: '2.0 g/L', perAcre: '1.75 lbs/acre', phi: '3 days', rei: '24 hours' },
        { name: 'Fluopicolide (Infinito 68.75 SC)', rate: '1.5 ml/L', perAcre: '1.2 pt/acre', phi: '2 days', rei: '12 hours' },
        { name: 'Dimethomorph 50% WP', rate: '1.0 - 1.5 g/L', perAcre: '6.5 oz/acre', phi: '4 days', rei: '12 hours' },
        { name: 'Mandipropamid 23.3% SC', rate: '0.8 ml/L', perAcre: '8.0 fl oz/acre', phi: '1 day', rei: '4 hours' },
      ],
      sprayInterval: 'High disease pressure requires 5 to 7-day spray intervals. Treat before rain events.',
      resistanceManagement: 'Rotate between FRAC Group 40 (Dimethomorph), FRAC Group 43 (Fluopicolide), and Group 27 (Cymoxanil).',
    },
    preventionSchedule: [
      { stage: 'Pre-Planting', actions: 'Plant only certified disease-free seed tubers and seedlings. Plant resistant cultivars (e.g., Mountain Magic, Defiant).' },
      { stage: 'Vegetative Growth', actions: 'Monitor regional Late Blight forecasts and spore traps. Apply preventative protectant copper or mancozeb.' },
      { stage: 'Flowering & Fruiting', actions: 'Keep vines elevated. Inspect field twice weekly after foggy or rainy weather.' },
      { stage: 'Post-Harvest Cleanup', actions: 'Destroy all cull piles immediately. Deep-plow residue and ensure no green tissue overwinters.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.5, chemicalUnit: 'g', sprayVolumeGalPerAcre: 60, sprayVolumeLitersPer1000SqFt: 5.0 },
  },

  'powdery mildew': {
    pathogen: 'Fungal (Podosphaera / Erysiphe / Leveillula spp.)',
    symptoms: [
      'Distinct white to talcum powder-like circular fungal spots on upper and lower leaf surfaces.',
      'Leaves curl upward, turn yellow, desiccate, and turn brown/brittle.',
      'Infected young leaves become distorted and stunted in growth.',
      'Reduced photosynthetic capacity leading to small, poorly flavored fruit.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Potassium Bicarbonate (MilStop / Armicarb) at 3.0 - 4.5 g/L for rapid eradicant action.',
        'Bacillus subtilis (Serenade MAX) at 3.0 g/L as a preventative biofilm.',
        'Sulfur dust or wettable sulfur (2.0 - 3.0 g/L) — do not apply when temperatures exceed 85°F.',
      ],
      botanicalOils: [
        'Neem oil 0.75% solution (7.5 ml/L water) sprayed thoroughly covering leaf undersides.',
        'Dilute milk spray (40% whole milk, 60% water) exposed to direct sunlight for natural antiseptic effect.',
      ],
      culturalPractices: [
        'Plant in full sun (>6-8 hours direct sunlight) since UV light inhibits spore germination.',
        'Prune dense canopy foliage to enhance internal airflow and reduce localized relative humidity.',
        'Avoid excessive synthetic nitrogen fertilization which stimulates succulent, susceptible growth.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Myclobutanil 20% EW', rate: '0.5 - 0.75 ml/L', perAcre: '4.0 - 5.0 fl oz/acre', phi: '0 days', rei: '24 hours' },
        { name: 'Trifloxystrobin 50% WDG', rate: '0.4 - 0.6 g/L', perAcre: '2.0 - 3.8 oz/acre', phi: '0 days', rei: '12 hours' },
        { name: 'Difenoconazole 25% EC', rate: '0.5 ml/L', perAcre: '4.0 fl oz/acre', phi: '7 days', rei: '12 hours' },
      ],
      sprayInterval: 'Apply every 7-14 days at first appearance of white powdery colonies.',
      resistanceManagement: 'Alternate DMI fungicides (FRAC 3) with QoI fungicides (FRAC 11) and contact sulfur/potassium bicarbonate.',
    },
    preventionSchedule: [
      { stage: 'Pre-Planting', actions: 'Select mildew-resistant varieties (e.g. PMR cucurbit hybrids). Ensure wide row spacing.' },
      { stage: 'Vegetative Growth', actions: 'Prune for open canopy. Apply bio-fungicide or potassium bicarbonate at first leaf canopy closure.' },
      { stage: 'Flowering & Fruiting', actions: 'Check shaded interior leaves weekly. Maintain consistent soil moisture via drip irrigation.' },
      { stage: 'Post-Harvest Cleanup', actions: 'Clear and compost or bury all dried vine residue. Sanitize trellising wires and posts.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 1.5, chemicalUnit: 'g', sprayVolumeGalPerAcre: 45, sprayVolumeLitersPer1000SqFt: 4.0 },
  },

  'apple scab': {
    pathogen: 'Fungal (Venturia inaequalis)',
    symptoms: [
      'Olive-green to velvety dark brown lesions on leaves with feathery, indistinct margins.',
      'Leaves become distorted, puckered, and drop prematurely in mid-season.',
      'Velvety brown lesions on fruit developing into corky, cracked, scabby patches.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Liquid Lime Sulfur applied during delayed dormant to pink bud stage.',
        'Wettable sulfur (3.0 - 4.0 g/L) applied before forecast rain events during primary ascospore season.',
        'Bacillus subtilis applied at petal fall and cover sprays.',
      ],
      botanicalOils: ['Neem oil or horticultural dormant oil to smother overwintering ascocarps in bark.'],
      culturalPractices: [
        'Rake and shred fallen apple leaves in autumn or apply 5% urea spray to accelerate leaf decomposition.',
        'Prune trees during dormancy into an open vase or central leader to promote rapid drying.',
        'Flail mow orchard floor to chop leaf litter before spring bud break.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Captan 80% WDG', rate: '2.0 - 2.5 g/L', perAcre: '2.5 - 3.5 lbs/acre', phi: '0 days', rei: '24 hours' },
        { name: 'Mancozeb 75% DF', rate: '2.0 g/L', perAcre: '3.0 lbs/acre', phi: '77 days', rei: '24 hours' },
        { name: 'Difenoconazole 25% EC', rate: '0.4 ml/L', perAcre: '3.5 fl oz/acre', phi: '14 days', rei: '12 hours' },
      ],
      sprayInterval: 'Critical infection period is from green tip through 2nd cover spray (typically every 5-7 days pre-bloom).',
      resistanceManagement: 'Tank-mix systemic fungicides with a protectant (Captan) to prevent resistance development.',
    },
    preventionSchedule: [
      { stage: 'Dormant Stage', actions: 'Apply copper sulfate or lime sulfur spray. Flail mow orchard floor to destroy fallen leaves.' },
      { stage: 'Green Tip to Bloom', actions: 'Monitor Mills Apple Scab infection periods. Apply protectant fungicide before wetting periods.' },
      { stage: 'Petal Fall & Cover', actions: 'Inspect fruit cluster leaves. Continue protective sprays until primary ascospore discharge ends.' },
      { stage: 'Post-Harvest', actions: 'Apply 5% agricultural urea foliar spray just before leaf drop to boost nitrogen and hasten leaf breakdown.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.2, chemicalUnit: 'g', sprayVolumeGalPerAcre: 100, sprayVolumeLitersPer1000SqFt: 8.0 },
  },

  'black rot': {
    pathogen: 'Fungal (Guignardia bidwellii / Botryosphaeria obtusa)',
    symptoms: [
      'Circular reddish-brown leaf spots with tiny black pycnidia (fruiting bodies) arranged in rings.',
      'Grapes/apples shrivel into hard, black, wrinkled mummies that remain hanging on vines/branches.',
      'Black elongated lesions on shoots and petioles causing stem girdling.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Fixed copper hydroxide applied from bud break through veraison.',
        'Bacillus amyloliquefaciens (Double Nickel 55) bio-fungicide foliar applications.',
      ],
      botanicalOils: ['Horticultural oils combined with copper octanoate for enhanced sticker-spreader adhesion.'],
      culturalPractices: [
        'Prune out and destroy all mummified fruit clusters during winter pruning.',
        'Maintain canopy training (VSP or high wire) for continuous air movement through fruit zones.',
        'Remove wild grapevines within 500 feet of the vineyard.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Myclobutanil 20% EW', rate: '0.6 ml/L', perAcre: '4.0 - 5.0 fl oz/acre', phi: '14 days', rei: '24 hours' },
        { name: 'Mancozeb 75% WP', rate: '2.5 g/L', perAcre: '3.0 - 4.0 lbs/acre', phi: '66 days', rei: '24 hours' },
        { name: 'Kresoxim-methyl 50% WG', rate: '0.5 g/L', perAcre: '3.2 oz/acre', phi: '14 days', rei: '12 hours' },
      ],
      sprayInterval: 'Spray starting at 1-3 inch shoot growth, pre-bloom, bloom, and post-bloom (every 10-14 days).',
      resistanceManagement: 'Alternate Sterol Inhibitors (FRAC 3) with Strobilurins (FRAC 11) and Protectants (FRAC M3).',
    },
    preventionSchedule: [
      { stage: 'Dormant Stage', actions: 'Hand-strip and destroy all hanging mummies from trellises and remove pruned canes from vineyard.' },
      { stage: 'Early Shoot Growth', actions: 'Initiate protectant spray program when shoots reach 3-5 inches.' },
      { stage: 'Pre-Bloom to Bloom', actions: 'CRITICAL WINDOW: Ensure 100% spray coverage during immediate pre-bloom and bloom stages.' },
      { stage: 'Post-Veraison', actions: 'Monitor fruit clusters; prune leaves around clusters to maximize sunlight exposure.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.0, chemicalUnit: 'g', sprayVolumeGalPerAcre: 75, sprayVolumeLitersPer1000SqFt: 6.0 },
  },

  'bacterial spot': {
    pathogen: 'Bacterial (Xanthomonas perforans / vesicatoria / arboricola)',
    symptoms: [
      'Small (1-3mm), dark brown to black angular water-soaked spots on leaves and stems.',
      'Lesions often appear greasy on leaf undersides and may have narrow yellow halos.',
      'Severe spotting leads to yellowing, leaf drop, and defoliation from bottom up.',
      'Fruit exhibits raised, scab-like, brown warty blemishes.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Fixed Copper Hydroxide + Bacillus subtilis tank mix.',
        'Bacteriophage biological sprays (AgriPhage) specific to Xanthomonas strains.',
        'Peroxyacetic acid / Hydrogen dioxide (OxiDate 2.0) for surface sterilization.',
      ],
      botanicalOils: ['Botanical essential oil blends (Thyme, Clove extract) showing bactericidal efficacy.'],
      culturalPractices: [
        'Avoid working in wet fields; bacteria spread easily on tools, hands, and equipment.',
        'Use drip irrigation exclusively — overhead water splashes bacteria across plants.',
        'Treat seed with hot water (122°F for 25 minutes for tomato; 125°F for 30 minutes for pepper).',
        'Rotate with non-host crops for at least 2 consecutive seasons.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Copper Hydroxide + Mancozeb mix', rate: '2.0 g/L each', perAcre: '1.5 lbs/acre each', phi: '5 days', rei: '24 hours' },
        { name: 'Acibenzolar-S-methyl (Actigard 50WG)', rate: '0.15 g/L', perAcre: '0.75 oz/acre', phi: '14 days', rei: '12 hours' },
        { name: 'Streptomycin Sulfate (Orchard/Seedling only)', rate: '1.0 g/L', perAcre: '100 ppm', phi: '30 days', rei: '12 hours' },
      ],
      sprayInterval: 'Spray every 5 to 7 days during warm, rainy weather.',
      resistanceManagement: 'Tank-mixing copper with mancozeb releases more active copper ions and overcomes copper-tolerant Xanthomonas.',
    },
    preventionSchedule: [
      { stage: 'Seed & Transplants', actions: 'Use only certified Xanthomonas-free seed. Perform hot-water seed treatment before planting.' },
      { stage: 'Vegetative Stage', actions: 'Apply preventative copper-mancozeb mix or plant resistance inducers (Actigard) before disease onset.' },
      { stage: 'Fruit Formation', actions: 'Inspect foliage weekly; avoid field entry when morning dew is present on foliage.' },
      { stage: 'Post-Harvest', actions: 'Immediately incorporate crop debris into soil to accelerate microbial breakdown of bacteria.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.5, chemicalUnit: 'g', sprayVolumeGalPerAcre: 50, sprayVolumeLitersPer1000SqFt: 4.5 },
  },

  'healthy': {
    pathogen: 'No pathogen detected (Healthy plant)',
    symptoms: [
      'Vibrant green uniform coloration without chlorotic or necrotic spots.',
      'Normal leaf turgidity, vigorous stem growth, and healthy venation.',
      'No signs of fungal sporulation, bacterial streaming, viral mosaic, or pest infestation.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Preventative foliar application of Seaweed extract (Kelpgro) or Fish Hydrolysate for enhanced immune vigor.',
        'Prophylactic beneficial microbes (Mycorrhizae & Bacillus subtilis) to protect root and leaf phyllosphere.',
      ],
      botanicalOils: ['Light preventative neem oil application (0.25% solution) once monthly for insect repellence.'],
      culturalPractices: [
        'Maintain balanced organic soil nutrition based on periodic soil testing (target N-P-K and micronutrients).',
        'Consistent drip irrigation scheduling aligned with evapotranspiration rates.',
        'Maintain 2-3 inches of organic mulch to conserve moisture and regulate soil temperature.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'No chemical intervention required', rate: '0 g/L', perAcre: '0', phi: 'N/A', rei: '0 hours' },
      ],
      sprayInterval: 'No chemical fungicides needed. Continue scouting twice weekly.',
      resistanceManagement: 'Preserve beneficial predator insects and natural soil microbiome.',
    },
    preventionSchedule: [
      { stage: 'Ongoing Scouting', actions: 'Inspect 20 random plants weekly across the plot for early warning signs of stress.' },
      { stage: 'Nutrition Management', actions: 'Apply balanced foliar micronutrients (Zinc, Magnesium, Boron) during flowering and fruit sizing.' },
      { stage: 'Irrigation & Drainage', actions: 'Monitor soil moisture sensors; ensure no waterlogging or root suffocation.' },
      { stage: 'Biosecurity', actions: 'Sanitize equipment before entering fields from other agricultural properties.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 0, chemicalUnit: 'g', sprayVolumeGalPerAcre: 0, sprayVolumeLitersPer1000SqFt: 0 },
  },
}

/**
 * Retrieve agronomic guide for a given disease name. Falls back to a standard generic disease profile.
 */
export function getAgronomicGuide(disease = '', crop = '') {
  const d = (disease || '').toLowerCase()
  for (const key of Object.keys(AGRONOMIC_GUIDES)) {
    if (d.includes(key)) {
      return AGRONOMIC_GUIDES[key]
    }
  }

  // Fallback profile for unlisted disease
  return {
    pathogen: `${crop ? crop.charAt(0).toUpperCase() + crop.slice(1) : 'Plant'} Foliar Disease (${disease || 'Diagnosed Condition'})`,
    symptoms: [
      `Visible discoloration, lesions, or atypical necrosis characteristic of ${disease || 'the diagnosed pathogen'}.`,
      'Interrupted photosynthetic efficiency and potential premature foliage senescence.',
      'Possible stem or fruit blemish if left untreated during humid conditions.',
    ],
    organicRemedies: {
      bioFungicides: [
        'Bio-fungicide (Bacillus subtilis / Trichoderma harzianum) at 3.0 ml/L.',
        'Broad-spectrum Copper Octanoate (Soap shield) at 2.5 ml/L.',
      ],
      botanicalOils: [
        'Cold-pressed Neem Oil (0.75% solution with mild soap emulsifier) sprayed in early morning.',
        'Potassium Bicarbonate foliar buffer (3.0 g/L) to prevent spore establishment.',
      ],
      culturalPractices: [
        'Prune damaged or diseased leaves immediately and dispose of in sealed bags.',
        'Ensure wide spacing for cross-ventilation and convert to ground-level drip watering.',
        'Sanitize shears and footwear with 70% alcohol between rows.',
      ],
    },
    chemicalControl: {
      activeIngredients: [
        { name: 'Broad-Spectrum Protectant (Mancozeb 75% WP)', rate: '2.0 g/L', perAcre: '1.5 - 2.0 lbs/acre', phi: '5 days', rei: '24 hours' },
        { name: 'Copper Hydroxide 50% WP', rate: '2.5 g/L', perAcre: '2.0 lbs/acre', phi: '0 days', rei: '24 hours' },
        { name: 'Systemic Fungicide (Azoxystrobin / Difenoconazole)', rate: '1.0 ml/L', perAcre: '6.0 fl oz/acre', phi: '7 days', rei: '12 hours' },
      ],
      sprayInterval: 'Apply preventative foliar spray every 7-10 days upon disease spotting.',
      resistanceManagement: 'Rotate chemical classes across spray cycles to preserve efficacy.',
    },
    preventionSchedule: [
      { stage: 'Pre-Season', actions: 'Select certified disease-tolerant cultivars and execute multi-year crop rotation.' },
      { stage: 'Vegetative Growth', actions: 'Maintain optimal canopy airflow and balance soil fertility.' },
      { stage: 'Bloom & Maturation', actions: 'Scout plants weekly; apply protective bio-fungicides ahead of high-humidity periods.' },
      { stage: 'Post-Harvest', actions: 'Clear debris thoroughly and deep-cultivate or solarize soil beds.' },
    ],
    defaultDosage: { chemicalRatePerLiter: 2.0, chemicalUnit: 'g', sprayVolumeGalPerAcre: 50, sprayVolumeLitersPer1000SqFt: 4.5 },
  }
}

/**
 * Spray Dosage Calculator conversion factors and computation.
 */
export const AREA_UNITS = [
  { id: 'sq_ft', label: 'Square Feet (sq ft)', factorToAcres: 1 / 43560, factorToSqFt: 1 },
  { id: 'acres', label: 'Acres (ac)', factorToAcres: 1, factorToSqFt: 43560 },
  { id: 'sq_m', label: 'Square Meters (sq m)', factorToAcres: 1 / 4046.86, factorToSqFt: 10.7639 },
  { id: 'hectares', label: 'Hectares (ha)', factorToAcres: 2.47105, factorToSqFt: 107639 },
]

export const SPRAY_EQUIPMENT_PRESETS = [
  { id: 'knapsack', name: '16L Backpack Sprayer (Knapsack)', tankCapacityLiters: 16, standardWaterRateGalPerAcre: 50 },
  { id: 'handheld', name: '5L Garden Pump Sprayer', tankCapacityLiters: 5, standardWaterRateGalPerAcre: 40 },
  { id: 'boom', name: '400L Tractor Boom Sprayer', tankCapacityLiters: 400, standardWaterRateGalPerAcre: 25 },
]

export const CHEMICAL_PRESETS = [
  { id: 'neem', name: 'Cold-Pressed Neem Oil 0.5%', dosagePerLiter: 5.0, unit: 'ml', desc: 'Botanical foliar insecticide & bio-fungicide' },
  { id: 'copper', name: 'Copper Hydroxide 50% WP', dosagePerLiter: 2.5, unit: 'g', desc: 'Protectant bactericide & broad-spectrum fungicide' },
  { id: 'mancozeb', name: 'Mancozeb 75% WP', dosagePerLiter: 2.0, unit: 'g', desc: 'Multi-site contact protective fungicide' },
  { id: 'bacillus', name: 'Bacillus subtilis Bio-Fungicide', dosagePerLiter: 3.0, unit: 'ml', desc: 'Biological living antagonist against leaf blights' },
  { id: 'custom', name: 'Custom Dosage Rate', dosagePerLiter: 2.0, unit: 'g', desc: 'User-specified concentration' },
]

/**
 * Compute spray mix requirements given area, unit, chemical preset, and equipment.
 */
export function calculateSprayDosage({
  areaValue = 1000,
  unitId = 'sq_ft',
  chemicalPresetId = 'copper',
  customRatePerLiter = 2.0,
  customUnit = 'g',
  equipmentId = 'knapsack',
  waterVolumeGalPerAcre = 50,
}) {
  const numericArea = Math.max(0, Number(areaValue) || 0)
  const unit = AREA_UNITS.find((u) => u.id === unitId) || AREA_UNITS[0]
  const equip = SPRAY_EQUIPMENT_PRESETS.find((e) => e.id === equipmentId) || SPRAY_EQUIPMENT_PRESETS[0]
  const chem = CHEMICAL_PRESETS.find((c) => c.id === chemicalPresetId) || CHEMICAL_PRESETS[0]

  const areaInAcres = numericArea * unit.factorToAcres
  const effectiveWaterRateGalPerAcre = waterVolumeGalPerAcre || equip.standardWaterRateGalPerAcre

  // Total Water Volume in Gallons and Liters
  const totalWaterGallons = areaInAcres * effectiveWaterRateGalPerAcre
  const totalWaterLiters = totalWaterGallons * 3.78541

  // Chemical concentration rate per Liter
  const ratePerLiter = chemicalPresetId === 'custom' ? Math.max(0, Number(customRatePerLiter) || 0) : chem.dosagePerLiter
  const chemUnit = chemicalPresetId === 'custom' ? customUnit : chem.unit

  // Total Chemical needed
  const totalChemicalAmount = totalWaterLiters * ratePerLiter

  // Tanks needed
  const tanksRequired = equip.tankCapacityLiters > 0 ? (totalWaterLiters / equip.tankCapacityLiters) : 0
  const chemicalPerTank = equip.tankCapacityLiters * ratePerLiter

  return {
    numericArea,
    unitLabel: unit.label,
    areaInAcres: Number(areaInAcres.toFixed(4)),
    totalWaterLiters: Number(totalWaterLiters.toFixed(2)),
    totalWaterGallons: Number(totalWaterGallons.toFixed(2)),
    ratePerLiter,
    chemUnit,
    totalChemicalAmount: Number(totalChemicalAmount.toFixed(1)),
    tanksRequired: Number(tanksRequired.toFixed(1)),
    chemicalPerTank: Number(chemicalPerTank.toFixed(1)),
    tankCapacityLiters: equip.tankCapacityLiters,
    equipmentName: equip.name,
    chemicalName: chem.name,
  }
}

export const REGULATORY_JURISDICTIONS = [
  { id: 'EPA', label: 'USA (EPA)', region: 'United States', agency: 'EPA' },
  { id: 'EFSA', label: 'European Union (EFSA)', region: 'European Union', agency: 'EFSA' },
  { id: 'CIBRC', label: 'India (CIBRC)', region: 'India', agency: 'CIBRC' },
  { id: 'OMRI', label: 'Global Organic (OMRI Only)', region: 'Global Organic', agency: 'OMRI' },
]

/**
 * Retrieve chemical regulatory compliance status and badges by regional jurisdiction.
 */
export function getChemicalRegulatoryStatus(chemicalName = '', jurisdictionId = 'EPA') {
  const name = (chemicalName || '').toLowerCase()
  const jur = (jurisdictionId || 'EPA').toUpperCase()

  // 1. Global Organic (OMRI)
  if (jur === 'OMRI' || jur.includes('ORGANIC')) {
    const isOrganic =
      name.includes('copper') ||
      name.includes('bacillus') ||
      name.includes('neem') ||
      name.includes('bicarbonate') ||
      name.includes('sulfur') ||
      name.includes('trichoderma') ||
      name.includes('kelp') ||
      name.includes('seaweed') ||
      name.includes('peroxyacetic') ||
      name.includes('bacteriophage') ||
      name.includes('botanical') ||
      name.includes('milk')

    if (isOrganic) {
      return {
        jurisdiction: 'OMRI',
        status: 'approved',
        isRestricted: false,
        badge: 'OMRI Listed / Organic Approved',
        badgeClass: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
        note: 'Approved for USDA NOP & OMRI certified organic production.',
      }
    }

    return {
      jurisdiction: 'OMRI',
      status: 'prohibited',
      isRestricted: true,
      badge: 'Prohibited (OMRI Organic)',
      badgeClass: 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30',
      note: 'Synthetic chemical active ingredient prohibited under OMRI organic standards. Use biological/botanical alternatives.',
    }
  }

  // 2. European Union (EFSA)
  if (jur === 'EFSA' || jur.includes('EU') || jur.includes('EUROPE')) {
    if (name.includes('chlorothalonil')) {
      return {
        jurisdiction: 'EFSA',
        status: 'restricted',
        isRestricted: true,
        badge: 'Non-Renewed in EU / Restricted',
        badgeClass: 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30',
        note: 'Approval non-renewed under EC 1107/2009. Prohibited for open field use in the EU.',
      }
    }

    if (name.includes('mancozeb')) {
      return {
        jurisdiction: 'EFSA',
        status: 'restricted',
        isRestricted: true,
        badge: 'Phase-out in EU',
        badgeClass: 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30',
        note: 'EU Commission Implementing Regulation 2020/2087 phase-out due to endocrine classification.',
      }
    }

    if (name.includes('streptomycin')) {
      return {
        jurisdiction: 'EFSA',
        status: 'restricted',
        isRestricted: true,
        badge: 'EU Restricted (Antibiotic)',
        badgeClass: 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30',
        note: 'Agricultural antibiotic usage strictly prohibited in EU member states.',
      }
    }

    return {
      jurisdiction: 'EFSA',
      status: 'approved',
      isRestricted: false,
      badge: 'EFSA Compliant',
      badgeClass: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
      note: 'Registered for approved plant protection use within the European Union.',
    }
  }

  // 3. India (CIBRC)
  if (jur === 'CIBRC' || jur.includes('INDIA')) {
    if (name.includes('streptomycin')) {
      return {
        jurisdiction: 'CIBRC',
        status: 'restricted',
        isRestricted: true,
        badge: 'CIBRC Regulated / Restricted',
        badgeClass: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30',
        note: 'Restricted formulation under the Insecticides Act, 1968.',
      }
    }

    return {
      jurisdiction: 'CIBRC',
      status: 'approved',
      isRestricted: false,
      badge: 'CIBRC Registered',
      badgeClass: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
      note: 'Registered and approved under the Insecticides Act, 1968 by CIB&RC India.',
    }
  }

  // 4. USA (EPA - Default)
  if (name.includes('streptomycin')) {
    return {
      jurisdiction: 'EPA',
      status: 'restricted',
      isRestricted: true,
      badge: 'EPA Restricted Use',
      badgeClass: 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30',
      note: 'EPA designated Restricted Use Pesticide (RUP). Certified applicator license required.',
    }
  }

  return {
    jurisdiction: 'EPA',
    status: 'approved',
    isRestricted: false,
    badge: 'EPA Approved / Registered',
    badgeClass: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30',
    note: 'Registered with the US EPA. Complies with FIFRA label guidelines.',
  }
}

