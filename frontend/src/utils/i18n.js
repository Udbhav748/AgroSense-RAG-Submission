/**
 * Multilingual Agricultural Localization & Vernacular Voice Dictionary.
 * Supports 6 major global agricultural languages:
 * - en: English (en-US)
 * - es: Spanish / Español (es-ES)
 * - hi: Hindi / हिन्दी (hi-IN)
 * - pt: Portuguese / Português (pt-BR)
 * - fr: French / Français (fr-FR)
 * - sw: Swahili / Kiswahili (sw-KE)
 */

export const SUPPORTED_LANGUAGES = [
  { code: 'en', label: 'English', nativeLabel: 'English', voiceLang: 'en-US' },
  { code: 'es', label: 'Spanish', nativeLabel: 'Español', voiceLang: 'es-ES' },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हिन्दी', voiceLang: 'hi-IN' },
  { code: 'pt', label: 'Portuguese', nativeLabel: 'Português', voiceLang: 'pt-BR' },
  { code: 'fr', label: 'French', nativeLabel: 'Français', voiceLang: 'fr-FR' },
  { code: 'sw', label: 'Swahili', nativeLabel: 'Kiswahili', voiceLang: 'sw-KE' },
]

export const VOICE_LANGUAGE_MAP = {
  en: 'en-US',
  es: 'es-ES',
  hi: 'hi-IN',
  pt: 'pt-BR',
  fr: 'fr-FR',
  sw: 'sw-KE',
}

export const TRANSLATIONS = {
  en: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: '24h Field Protocol',
      symptoms: 'Symptoms & Visuals',
      organic: 'Organic & Bio Remedies',
      chemical: 'Chemical Controls',
      cultural: 'Long-Term Practices',
      citations: 'Extension Citations',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'Diagnose Plant Leaf',
      analyzingLeaf: 'Analyzing Leaf...',
      chooseAnotherPhoto: 'Choose Another Photo',
      listenProtocol: '[ Listen to 24h Field Protocol ]',
      resume: 'Resume',
      pause: 'Pause',
      stop: 'Stop Audio',
      voiceDictation: 'Hands-free Field Voice Dictation',
      listening: 'Listening... Speak clearly',
      continueChat: 'Continue in Agronomic Chat',
      downloadPrescription: 'Download Spray Prescription Work Order',
      saveToLog: 'Save to Field Scouting Log',
      savedToLog: 'Saved to Field Log',
      openCalculator: 'Open Spray Calculator',
      hideCalculator: 'Hide Calculator',
      tryAgain: 'Try Again',
      selectLanguage: 'Select Language',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'Visual Diagnosis',
      pathogen: 'Pathogen',
      severity: 'Severity',
      severityLow: 'Low',
      severityModerate: 'Moderate',
      severitySevere: 'Severe',
      organicRemedies: 'Organic & Biological Remedies',
      biologicalControl: 'Biological Control',
      culturalPractices: 'Cultural Practices & Sanitation',
      chemicalControl: 'Chemical Control',
      activeIngredient: 'Active Ingredient',
      fracCode: 'FRAC Code',
      dosageProtocol: 'Dosage Protocol',
      sprayInterval: 'Spray Interval',
      preHarvestInterval: 'Pre-Harvest Interval (PHI)',
      reEntryInterval: 'Re-Entry Interval (REI)',
      ppeRequirements: 'Personal Protective Equipment (PPE)',
      fieldSanitation: 'Field Sanitation',
      cropRotation: 'Crop Rotation',
      dripIrrigation: 'Drip Irrigation',
      balancedNutrition: 'Balanced Nutrition',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: '24h Field Emergency Protocol Audio',
      subtitle: 'Emergency 24-48h action protocol narrated for noisy outdoor conditions',
      noDiagnosis: 'No active crop diagnosis available.',
      healthyReport: 'Field Health Report: Your {crop} foliage is diagnosed as healthy with {confidence}% certainty. Severity is low. Continue routine weekly scouting and balanced organic nutrition. No chemical fungicide is required.',
      emergencyIntro: 'Emergency 24 to 48-Hour Field Protocol for {crop} affected by {disease}.',
      severityLevel: 'Severity level: {level} with {confidence}% visual certainty.',
      pathogenLabel: 'Pathogen: {pathogen}.',
      step1Sanitation: 'Immediate Step 1: Cultural Sanitation. {step}',
      step2Biological: 'Immediate Step 2: Biological Control. {step}',
      step3Chemical: 'Immediate Step 3: Chemical Treatment if infection persists. Apply {step}.',
      sprayIntervalLabel: 'Spray interval: {interval}',
      safetyAdvisory: 'Safety Advisory: Always wear personal protective equipment and observe pre-harvest interval regulations.',
    },
    // Dosage Units
    units: {
      g_per_l: 'g/L',
      ml_per_l: 'mL/L',
      kg_per_ha: 'kg/ha',
      l_per_ha: 'L/ha',
      oz_per_gal: 'oz/gal',
      fl_oz_per_gal: 'fl oz/gal',
      lb_per_acre: 'lb/acre',
      gal_per_acre: 'gal/acre',
      ppm: 'ppm',
    },
  },
  es: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: 'Protocolo de Campo 24h',
      symptoms: 'Síntomas y Diagnóstico Visual',
      organic: 'Remedios Orgánicos y Biológicos',
      chemical: 'Control Químico',
      cultural: 'Prácticas Culturales a Largo Plazo',
      citations: 'Citas de Extensión Universitaria',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'Diagnosticar Hoja',
      analyzingLeaf: 'Analizando Hoja...',
      chooseAnotherPhoto: 'Elegir Otra Foto',
      listenProtocol: '[ Escuchar Protocolo de Campo 24h ]',
      resume: 'Reanudar',
      pause: 'Pausar',
      stop: 'Detener Audio',
      voiceDictation: 'Dictado por Voz en Campo (Manos Libres)',
      listening: 'Escuchando... Hable claramente',
      continueChat: 'Continuar en Chat Agronómico',
      downloadPrescription: 'Descargar Orden de Receta de Pulverización',
      saveToLog: 'Guardar en Registro de Campo',
      savedToLog: 'Guardado en Registro',
      openCalculator: 'Abrir Calculadora de Pulverización',
      hideCalculator: 'Ocultar Calculadora',
      tryAgain: 'Intentar de Nuevo',
      selectLanguage: 'Seleccionar Idioma',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'Diagnóstico Visual',
      pathogen: 'Patógeno',
      severity: 'Severidad',
      severityLow: 'Baja',
      severityModerate: 'Moderada',
      severitySevere: 'Severa',
      organicRemedies: 'Remedios Orgánicos y Biológicos',
      biologicalControl: 'Control Biológico',
      culturalPractices: 'Prácticas Culturales y Sanidad',
      chemicalControl: 'Control Químico',
      activeIngredient: 'Ingrediente Activo',
      fracCode: 'Código FRAC',
      dosageProtocol: 'Protocolo de Dosis',
      sprayInterval: 'Intervalo de Pulverización',
      preHarvestInterval: 'Intervalo Pre-Cosecha (PHI)',
      reEntryInterval: 'Intervalo de Reentrada (REI)',
      ppeRequirements: 'Equipo de Protección Personal (EPP)',
      fieldSanitation: 'Sanidad de Campo',
      cropRotation: 'Rotación de Cultivos',
      dripIrrigation: 'Riego por Goteo',
      balancedNutrition: 'Nutrición Equilibrada',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: 'Audio del Protocolo de Emergencia 24h',
      subtitle: 'Protocolo de acción de emergencia 24-48h narrado para condiciones de campo',
      noDiagnosis: 'No hay diagnóstico de cultivo activo disponible.',
      healthyReport: 'Informe de Salud de Campo: El follaje de su {crop} se diagnostica como saludable con un {confidence}% de certeza. La severidad es baja. Continúe el monitoreo semanal rutinario y la nutrición orgánica equilibrada. No se requiere fungicida químico.',
      emergencyIntro: 'Protocolo de Campo de Emergencia de 24 a 48 Horas para {crop} afectado por {disease}.',
      severityLevel: 'Nivel de severidad: {level} con {confidence}% de certeza visual.',
      pathogenLabel: 'Patógeno: {pathogen}.',
      step1Sanitation: 'Paso Inmediato 1: Saneamiento Cultural. {step}',
      step2Biological: 'Paso Inmediato 2: Control Biológico. {step}',
      step3Chemical: 'Paso Inmediato 3: Tratamiento Químico si la infección persiste. Aplicar {step}.',
      sprayIntervalLabel: 'Intervalo de pulverización: {interval}',
      safetyAdvisory: 'Aviso de Seguridad: Use siempre equipo de protección personal y respete los intervalos de seguridad pre-cosecha.',
    },
    // Dosage Units
    units: {
      g_per_l: 'g/L',
      ml_per_l: 'mL/L',
      kg_per_ha: 'kg/ha',
      l_per_ha: 'L/ha',
      oz_per_gal: 'oz/gal',
      fl_oz_per_gal: 'fl oz/gal',
      lb_per_acre: 'lb/acre',
      gal_per_acre: 'gal/acre',
      ppm: 'ppm',
    },
  },
  hi: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: '24 घंटे का फील्ड प्रोटोकॉल',
      symptoms: 'लक्षण और दृश्य निदान',
      organic: 'जैविक और बायो उपचार',
      chemical: 'रासायनिक नियंत्रण',
      cultural: 'दीर्घकालिक कृषि पद्धतियां',
      citations: 'विस्तार अनुसंधान संदर्भ',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'पत्ती का निदान करें',
      analyzingLeaf: 'पत्ती का विश्लेषण हो रहा है...',
      chooseAnotherPhoto: 'दूसरी फोटो चुनें',
      listenProtocol: '[ 24 घंटे का फील्ड प्रोटोकॉल सुनें ]',
      resume: 'पुनः शुरू करें',
      pause: 'रोकें',
      stop: 'ऑडियो बंद करें',
      voiceDictation: 'हैंड्स-फ्री वॉयस डिक्टेशन',
      listening: 'सुन रहा है... स्पष्ट बोलें',
      continueChat: 'कृषि चैट में जारी रखें',
      downloadPrescription: 'स्प्रे नुस्खा कार्य आदेश डाउनलोड करें',
      saveToLog: 'फील्ड स्काउटिंग लॉग में सहेजें',
      savedToLog: 'लॉग में सहेजा गया',
      openCalculator: 'स्प्रे कैलकुलेटर खोलें',
      hideCalculator: 'कैलकुलेटर छिपाएं',
      tryAgain: 'पुनः प्रयास करें',
      selectLanguage: 'भाषा चुनें',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'दृश्य निदान',
      pathogen: 'रोगज़नक़ (Pathogen)',
      severity: 'गंभीरता (Severity)',
      severityLow: 'कम (Low)',
      severityModerate: 'मध्यम (Moderate)',
      severitySevere: 'गंभीर (Severe)',
      organicRemedies: 'जैविक और बायो उपचार',
      biologicalControl: 'जैविक नियंत्रण',
      culturalPractices: 'सांस्कृतिक पद्धतियां और स्वच्छता',
      chemicalControl: 'रासायनिक नियंत्रण',
      activeIngredient: 'सक्रिय घटक (Active Ingredient)',
      fracCode: 'FRAC कोड',
      dosageProtocol: 'खुराक प्रोटोकॉल',
      sprayInterval: 'छिड़काव अंतराल',
      preHarvestInterval: 'कटाई पूर्व अंतराल (PHI)',
      reEntryInterval: 'पुनः प्रवेश अंतराल (REI)',
      ppeRequirements: 'व्यक्तिगत सुरक्षा उपकरण (PPE)',
      fieldSanitation: 'खेत की स्वच्छता',
      cropRotation: 'फसल चक्र',
      dripIrrigation: 'टपक सिंचाई (Drip Irrigation)',
      balancedNutrition: 'संतुलित पोषण',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: '24 घंटे का आपातकालीन फील्ड प्रोटोकॉल ऑडियो',
      subtitle: 'खेत की परिस्थितियों के लिए 24-48 घंटे का आपातकालीन प्रोटोकॉल',
      noDiagnosis: 'कोई सक्रिय फसल निदान उपलब्ध नहीं है।',
      healthyReport: 'फील्ड स्वास्थ्य रिपोर्ट: आपकी {crop} की पत्तियां {confidence}% निश्चितता के साथ स्वस्थ पाई गई हैं। गंभीरता कम है। नियमित साप्ताहिक निरीक्षण और संतुलित जैविक पोषण जारी रखें। किसी रासायनिक कवकनाशी की आवश्यकता नहीं है।',
      emergencyIntro: '{disease} से प्रभावित {crop} के लिए आपातकालीन 24 से 48 घंटे का फील्ड प्रोटोकॉल।',
      severityLevel: 'गंभीरता स्तर: {level}, {confidence}% दृश्य निश्चितता के साथ।',
      pathogenLabel: 'रोगज़नक़: {pathogen}।',
      step1Sanitation: 'तत्काल चरण 1: सांस्कृतिक स्वच्छता। {step}',
      step2Biological: 'तत्काल चरण 2: जैविक नियंत्रण। {step}',
      step3Chemical: 'तत्काल चरण 3: संक्रमण जारी रहने पर रासायनिक उपचार। {step} लागू करें।',
      sprayIntervalLabel: 'छिड़काव अंतराल: {interval}',
      safetyAdvisory: 'सुरक्षा सलाह: हमेशा व्यक्तिगत सुरक्षा उपकरण (PPE) पहनें और कटाई पूर्व अंतराल नियमों का पालन करें।',
    },
    // Dosage Units
    units: {
      g_per_l: 'ग्राम/लीटर',
      ml_per_l: 'मिली/लीटर',
      kg_per_ha: 'किग्रा/हेक्टेयर',
      l_per_ha: 'लीटर/हेक्टेयर',
      oz_per_gal: 'औंस/गैलन',
      fl_oz_per_gal: 'द्रव औंस/गैलन',
      lb_per_acre: 'पाउंड/एकड़',
      gal_per_acre: 'गैलन/एकड़',
      ppm: 'ppm',
    },
  },
  pt: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: 'Protocolo de Campo 24h',
      symptoms: 'Sintomas e Diagnóstico Visual',
      organic: 'Remédios Orgânicos e Biológicos',
      chemical: 'Controle Químico',
      cultural: 'Práticas Culturais de Longo Prazo',
      citations: 'Citações de Extensão Universitária',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'Diagnosticar Folha',
      analyzingLeaf: 'Analisando Folha...',
      chooseAnotherPhoto: 'Escolher Outra Foto',
      listenProtocol: '[ Ouvir Protocolo de Campo 24h ]',
      resume: 'Retomar',
      pause: 'Pausar',
      stop: 'Parar Áudio',
      voiceDictation: 'Ditado por Voz no Campo (Mãos Livres)',
      listening: 'Ouvindo... Fale com clareza',
      continueChat: 'Continuar no Chat Agronômico',
      downloadPrescription: 'Baixar Ordem de Receituário de Pulverização',
      saveToLog: 'Salvar no Registro de Campo',
      savedToLog: 'Salvo no Registro',
      openCalculator: 'Abrir Calculadora de Pulverização',
      hideCalculator: 'Ocultar Calculadora',
      tryAgain: 'Tentar Novamente',
      selectLanguage: 'Selecionar Idioma',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'Diagnóstico Visual',
      pathogen: 'Patógeno',
      severity: 'Severidade',
      severityLow: 'Baixa',
      severityModerate: 'Moderada',
      severitySevere: 'Severa',
      organicRemedies: 'Remédios Orgânicos e Biológicos',
      biologicalControl: 'Controle Biológico',
      culturalPractices: 'Práticas Culturais e Sanitização',
      chemicalControl: 'Controle Químico',
      activeIngredient: 'Ingrediente Ativo',
      fracCode: 'Código FRAC',
      dosageProtocol: 'Protocolo de Dosagem',
      sprayInterval: 'Intervalo de Pulverização',
      preHarvestInterval: 'Intervalo de Segurança / Pré-Colheita (PHI)',
      reEntryInterval: 'Intervalo de Reentrada (REI)',
      ppeRequirements: 'Equipamento de Proteção Individual (EPI)',
      fieldSanitation: 'Sanitização do Campo',
      cropRotation: 'Rotação de Culturas',
      dripIrrigation: 'Irrigação por Gotejamento',
      balancedNutrition: 'Nutrição Equilibrada',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: 'Áudio do Protocolo de Emergência 24h',
      subtitle: 'Protocolo de ação de emergência 24-48h narrado para condições de campo',
      noDiagnosis: 'Nenhum diagnóstico de cultura ativo disponível.',
      healthyReport: 'Relatório de Saúde de Campo: A folhagem de {crop} foi diagnosticada como saudável com {confidence}% de certeza. A severidade é baixa. Continue o monitoramento semanal rotineiro e nutrição orgânica equilibrada. Nenhum fungicida químico é necessário.',
      emergencyIntro: 'Protocolo de Emergência de 24 a 48 Horas para {crop} afetado por {disease}.',
      severityLevel: 'Nível de severidade: {level} com {confidence}% de certeza visual.',
      pathogenLabel: 'Patógeno: {pathogen}.',
      step1Sanitation: 'Passo Imediato 1: Sanitização Cultural. {step}',
      step2Biological: 'Passo Imediato 2: Controle Biológico. {step}',
      step3Chemical: 'Passo Imediato 3: Tratamento Químico se a infecção persistir. Aplicar {step}.',
      sprayIntervalLabel: 'Intervalo de pulverização: {interval}',
      safetyAdvisory: 'Aviso de Segurança: Sempre use equipamento de proteção individual e observe os períodos de carência pré-colheita.',
    },
    // Dosage Units
    units: {
      g_per_l: 'g/L',
      ml_per_l: 'mL/L',
      kg_per_ha: 'kg/ha',
      l_per_ha: 'L/ha',
      oz_per_gal: 'oz/gal',
      fl_oz_per_gal: 'fl oz/gal',
      lb_per_acre: 'lb/acre',
      gal_per_acre: 'gal/acre',
      ppm: 'ppm',
    },
  },
  fr: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: 'Protocole de Terrain 24h',
      symptoms: 'Symptômes et Diagnostic Visuel',
      organic: 'Remèdes Biologiques et Naturels',
      chemical: 'Traitements Chimiques',
      cultural: 'Pratiques Culturales Durables',
      citations: 'Références de Recherche Agricole',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'Diagnostiquer la Feuille',
      analyzingLeaf: 'Analyse de la Feuille...',
      chooseAnotherPhoto: 'Choisir une Autre Photo',
      listenProtocol: '[ Écouter le Protocole 24h ]',
      resume: 'Reprendre',
      pause: 'Pause',
      stop: 'Arrêter l\'Audio',
      voiceDictation: 'Dictée Vocale Mains Libres',
      listening: 'Écoute en cours... Parlez clairement',
      continueChat: 'Continuer dans le Chat Agronomique',
      downloadPrescription: 'Télécharger l\'Ordonnance de Traitement',
      saveToLog: 'Enregistrer dans le Journal de Terrain',
      savedToLog: 'Enregistré dans le Journal',
      openCalculator: 'Ouvrir le Calculateur de Dosage',
      hideCalculator: 'Masquer le Calculateur',
      tryAgain: 'Réessayer',
      selectLanguage: 'Sélectionner la Langue',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'Diagnostic Visuel',
      pathogen: 'Agent Pathogène',
      severity: 'Sévérité',
      severityLow: 'Faible',
      severityModerate: 'Modérée',
      severitySevere: 'Sévère',
      organicRemedies: 'Remèdes Biologiques et Naturels',
      biologicalControl: 'Lutte Biologique',
      culturalPractices: 'Pratiques Culturales et Prophylaxie',
      chemicalControl: 'Lutte Chimique',
      activeIngredient: 'Matière Active',
      fracCode: 'Code FRAC',
      dosageProtocol: 'Protocole de Dosage',
      sprayInterval: 'Intervalle de Pulvérisation',
      preHarvestInterval: 'Délai Avant Récolte (DAR / PHI)',
      reEntryInterval: 'Délai de Réentrée (DRE / REI)',
      ppeRequirements: 'Équipement de Protection Individuelle (EPI)',
      fieldSanitation: 'Assainissement des Parcelles',
      cropRotation: 'Rotation des Cultures',
      dripIrrigation: 'Goutte-à-Goutte',
      balancedNutrition: 'Nutrition Équilibrée',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: 'Audio du Protocole d\'Urgence 24h',
      subtitle: 'Protocole d\'action d\'urgence 24-48h narré pour les conditions extérieures',
      noDiagnosis: 'Aucun diagnostic de culture actif disponible.',
      healthyReport: 'Rapport Sanitaire de Terrain : Le feuillage de votre {crop} est diagnostiqué sain avec {confidence}% de certitude. La sévérité est faible. Poursuivez la surveillance hebdomadaire et une fertilisation biologique équilibrée. Aucun fongicide chimique n\'est requis.',
      emergencyIntro: 'Protocole de Terrain d\'Urgence 24 à 48 Heures pour {crop} atteint de {disease}.',
      severityLevel: 'Niveau de sévérité : {level} avec {confidence}% de certitude visuelle.',
      pathogenLabel: 'Pathogène : {pathogen}.',
      step1Sanitation: 'Étape Immédiate 1 : Mesures Prophylactiques. {step}',
      step2Biological: 'Étape Immédiate 2 : Contrôle Biologique. {step}',
      step3Chemical: 'Étape Immédiate 3 : Traitement Chimique si l\'infection persiste. Appliquer {step}.',
      sprayIntervalLabel: 'Intervalle de pulvérisation : {interval}',
      safetyAdvisory: 'Avis de Sécurité : Portez toujours un équipement de protection individuelle et respectez les délais avant récolte.',
    },
    // Dosage Units
    units: {
      g_per_l: 'g/L',
      ml_per_l: 'mL/L',
      kg_per_ha: 'kg/ha',
      l_per_ha: 'L/ha',
      oz_per_gal: 'oz/gal',
      fl_oz_per_gal: 'fl oz/gal',
      lb_per_acre: 'lb/acre',
      gal_per_acre: 'gal/acre',
      ppm: 'ppm',
    },
  },
  sw: {
    // UI Tab Labels
    tabs: {
      fieldProtocol: 'Itifaki ya Shamba ya Saa 24',
      symptoms: 'Dalili na Utambuzi wa Macho',
      organic: 'Dawa za Asili na Kibiolojia',
      chemical: 'Udhibiti wa Kikemikali',
      cultural: 'Mbinu Endelevu za Kilimo',
      citations: 'Marejeleo ya Utafiti wa Kilimo',
    },
    // Button & Action Labels
    buttons: {
      diagnoseLeaf: 'Chunguza Jani la Mmea',
      analyzingLeaf: 'Inachunguza Jani...',
      chooseAnotherPhoto: 'Chagua Picha Nyingine',
      listenProtocol: '[ Sikiliza Itifaki ya Saa 24 ]',
      resume: 'Endelea',
      pause: 'Simamisha',
      stop: 'Zima Sauti',
      voiceDictation: 'Kuamuru kwa Sauti Shambani',
      listening: 'Inasikiliza... Zungumza wazi',
      continueChat: 'Endelea kwenye Mazungumzo ya Kilimo',
      downloadPrescription: 'Pakua Maagizo ya Upuliziaji Dawa',
      saveToLog: 'Hifadhi kwenye Kumbukumbu ya Shamba',
      savedToLog: 'Imehifadhiwa Shambani',
      openCalculator: 'Fungua Kikokotoo cha Dawa',
      hideCalculator: 'Ficha Kikokotoo',
      tryAgain: 'Jaribu Tena',
      selectLanguage: 'Chagua Lugha',
    },
    // Agronomic Terms
    agronomy: {
      visualDiagnosis: 'Utambuzi wa Macho',
      pathogen: 'Kimelea cha Ugonjwa',
      severity: 'Kiwango cha Ukali',
      severityLow: 'Chini',
      severityModerate: 'Wastani',
      severitySevere: 'Kikali',
      organicRemedies: 'Dawa za Asili na Kibiolojia',
      biologicalControl: 'Udhibiti wa Kibiolojia',
      culturalPractices: 'Mbinu za Kilimo na Usafi wa Shamba',
      chemicalControl: 'Udhibiti wa Kikemikali',
      activeIngredient: 'Kiambato Amilifu',
      fracCode: 'Msimbo wa FRAC',
      dosageProtocol: 'Kiwango cha Kipimo',
      sprayInterval: 'Muda wa Kurudia Kupuliza',
      preHarvestInterval: 'Muda Kabla ya Kuvuna (PHI)',
      reEntryInterval: 'Muda Kabla ya Kuingia Shambani (REI)',
      ppeRequirements: 'Vifaa vya Kujikinga (PPE)',
      fieldSanitation: 'Usafi wa Shamba',
      cropRotation: 'Mzunguko wa Mazao',
      dripIrrigation: 'Umwagiliaji wa Matone',
      balancedNutrition: 'Lishe Bora ya Mimea',
    },
    // Emergency Protocol Headings & Audio Script Components
    protocol: {
      title: 'Sauti ya Itifaki ya Dharura ya Saa 24',
      subtitle: 'Itifaki ya dharura ya saa 24-48 inayotamkwa kwa mazingira ya shamba',
      noDiagnosis: 'Hakuna utambuzi wa zao uliopo kwa sasa.',
      healthyReport: 'Ripoti ya Afya ya Shamba: Majani ya {crop} yako yana afya nzuri kwa uhakika wa {confidence}%. Ukali ni wa chini. Endelea na ukaguzi wa kila wiki na lishe bora ya asili. Hakuna dawa ya ukungu inayohitajika.',
      emergencyIntro: 'Itifaki ya Dharura ya Saa 24 hadi 48 kwa {crop} iliyoathiriwa na {disease}.',
      severityLevel: 'Kiwango cha ukali: {level} kwa uhakika wa kuona wa {confidence}%.',
      pathogenLabel: 'Kimelea: {pathogen}.',
      step1Sanitation: 'Hatua ya Haraka 1: Usafi wa Shamba. {step}',
      step2Biological: 'Hatua ya Haraka 2: Udhibiti wa Kibiolojia. {step}',
      step3Chemical: 'Hatua ya Haraka 3: Matibabu ya Kikemikali maambukizi yakiendelea. Tumia {step}.',
      sprayIntervalLabel: 'Muda wa kupuliza: {interval}',
      safetyAdvisory: 'Ushauri wa Usalama: Vaa vifaa vya kujikinga kila wakati na uzingatie muda uliowekwa kabla ya kuvuna.',
    },
    // Dosage Units
    units: {
      g_per_l: 'g/L',
      ml_per_l: 'mL/L',
      kg_per_ha: 'kg/ha',
      l_per_ha: 'L/ha',
      oz_per_gal: 'oz/gal',
      fl_oz_per_gal: 'fl oz/gal',
      lb_per_acre: 'lb/ekari',
      gal_per_acre: 'gal/ekari',
      ppm: 'ppm',
    },
  },
}

/**
 * Retrieve voice language code for Web Speech API (STT & TTS).
 * @param {string} lang
 * @returns {string} e.g. 'en-US', 'es-ES', 'hi-IN', 'pt-BR', 'fr-FR', 'sw-KE'
 */
export function getVoiceLanguage(lang = 'en') {
  const code = (lang || 'en').toLowerCase().split('-')[0]
  return VOICE_LANGUAGE_MAP[code] || 'en-US'
}

/**
 * Retrieve language configuration object.
 * @param {string} lang
 * @returns {{ code: string, label: string, nativeLabel: string, voiceLang: string }}
 */
export function getLanguageConfig(lang = 'en') {
  const code = (lang || 'en').toLowerCase().split('-')[0]
  return SUPPORTED_LANGUAGES.find((item) => item.code === code) || SUPPORTED_LANGUAGES[0]
}

/**
 * Interpolates string with object parameters: "Hello {name}" -> "Hello John"
 */
function interpolate(template, params = {}) {
  if (typeof template !== 'string') return ''
  return template.replace(/\{(\w+)\}/g, (match, key) => {
    return params[key] !== undefined ? params[key] : match
  })
}

/**
 * Get translated text string by dot-path key with fallback to English.
 * @param {string} path e.g. 'tabs.fieldProtocol', 'buttons.diagnoseLeaf', 'protocol.emergencyIntro'
 * @param {string} [lang='en']
 * @param {object} [params={}]
 * @returns {string}
 */
export function t(path, lang = 'en', params = {}) {
  const code = (lang || 'en').toLowerCase().split('-')[0]
  const dict = TRANSLATIONS[code] || TRANSLATIONS.en
  const enDict = TRANSLATIONS.en

  const parts = path.split('.')
  let current = dict
  let enCurrent = enDict

  for (const part of parts) {
    current = current?.[part]
    enCurrent = enCurrent?.[part]
  }

  const result = typeof current === 'string' ? current : typeof enCurrent === 'string' ? enCurrent : path
  return interpolate(result, params)
}

/**
 * Get translated agronomic term.
 * @param {string} term
 * @param {string} [lang='en']
 * @returns {string}
 */
export function getAgronomicTerm(term, lang = 'en') {
  return t(`agronomy.${term}`, lang)
}
