import React, { useState } from 'react';
import { FileText, Search, MapPin, Bug, Wheat, Calendar } from 'lucide-react';

interface NERAnalysisProps {
  language: string;
}

interface Entity {
  text: string;
  type: string;
  confidence: number;
  start: number;
  end: number;
}

export const NERAnalysis: React.FC<NERAnalysisProps> = ({ language }) => {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [entities, setEntities] = useState<Entity[]>([]);

  const translations = {
    en: {
      title: 'Named Entity Recognition',
      subtitle: 'Extract agricultural entities from Hindi and English text',
      placeholder: 'Enter text in Hindi or English to analyze agricultural entities...',
      analyze: 'Analyze Text',
      results: 'Extracted Entities',
      sampleText: 'Sample Text',
      crop: 'Crop',
      location: 'Location',
      pest: 'Pest/Disease',
      date: 'Date/Time',
      noEntities: 'No entities found in the text',
      confidence: 'Confidence',
      examples: 'Example Texts'
    },
    hi: {
      title: 'नामित इकाई पहचान',
      subtitle: 'हिंदी और अंग्रेजी टेक्स्ट से कृषि संस्थाओं को निकालें',
      placeholder: 'कृषि संस्थाओं का विश्लेषण करने के लिए हिंदी या अंग्रेजी में टेक्स्ट दर्ज करें...',
      analyze: 'टेक्स्ट का विश्लेषण करें',
      results: 'निकाली गई इकाइयां',
      sampleText: 'नमूना टेक्स्ट',
      crop: 'फसल',
      location: 'स्थान',
      pest: 'कीट/रोग',
      date: 'दिनांक/समय',
      noEntities: 'टेक्स्ट में कोई इकाई नहीं मिली',
      confidence: 'विश्वास',
      examples: 'उदाहरण टेक्स्ट'
    }
  };

  const t = translations[language as keyof typeof translations];

  const sampleTexts = {
    en: [
      "The wheat crop in Punjab is affected by stem rust. Farmers in Ludhiana district are reporting significant yield losses in March 2024.",
      "Tomato cultivation in Maharashtra faces challenges due to late blight disease. The affected areas include Pune and Nashik regions.",
      "Rice farmers in West Bengal are dealing with brown plant hopper infestations during the monsoon season."
    ],
    hi: [
      "पंजाब में गेहूं की फसल तना रस्ट से प्रभावित है। लुधियाना जिले के किसान मार्च 2024 में महत्वपूर्ण उपज हानि की रिपोर्ट कर रहे हैं।",
      "महाराष्ट्र में टमाटर की खेती को झुलसा रोग के कारण चुनौतियों का सामना करना पड़ रहा है। प्रभावित क्षेत्रों में पुणे और नासिक क्षेत्र शामिल हैं।",
      "पश्चिम बंगाल के धान किसान मानसून के मौसम में भूरे पौधे फुदके के संक्रमण से निपट रहे हैं।"
    ]
  };

  const mockNER = (text: string): Entity[] => {
    const entities: Entity[] = [];
    
    // Mock entity recognition patterns
    const patterns = {
      crop: {
        en: ['wheat', 'rice', 'tomato', 'corn', 'barley', 'cotton', 'sugarcane'],
        hi: ['गेहूं', 'धान', 'टमाटर', 'मक्का', 'जौ', 'कपास', 'गन्ना']
      },
      location: {
        en: ['Punjab', 'Maharashtra', 'West Bengal', 'Ludhiana', 'Pune', 'Nashik'],
        hi: ['पंजाब', 'महाराष्ट्र', 'पश्चिम बंगाल', 'लुधियाना', 'पुणे', 'नासिक']
      },
      pest: {
        en: ['stem rust', 'late blight', 'brown plant hopper', 'aphids', 'bollworm'],
        hi: ['तना रस्ट', 'झुलसा रोग', 'भूरा पौधा फुदका', 'माहू', 'सुंडी']
      },
      date: {
        en: ['March 2024', 'monsoon season', 'summer', 'winter', 'January'],
        hi: ['मार्च 2024', 'मानसून का मौसम', 'गर्मी', 'सर्दी', 'जनवरी']
      }
    };

    // Simple pattern matching for demo
    Object.entries(patterns).forEach(([type, langPatterns]) => {
      const allPatterns = [...langPatterns.en, ...langPatterns.hi];
      allPatterns.forEach(pattern => {
        const regex = new RegExp(pattern, 'gi');
        let match;
        while ((match = regex.exec(text)) !== null) {
          entities.push({
            text: match[0],
            type,
            confidence: Math.random() * 0.3 + 0.7, // 70-100%
            start: match.index,
            end: match.index + match[0].length
          });
        }
      });
    });

    return entities.sort((a, b) => a.start - b.start);
  };

  const analyzeText = () => {
    if (!inputText.trim()) return;
    
    setIsAnalyzing(true);
    
    // Simulate processing time
    setTimeout(() => {
      const extractedEntities = mockNER(inputText);
      setEntities(extractedEntities);
      setIsAnalyzing(false);
    }, 2000);
  };

  const getEntityIcon = (type: string) => {
    switch (type) {
      case 'crop':
        return <Wheat className="h-4 w-4" />;
      case 'location':
        return <MapPin className="h-4 w-4" />;
      case 'pest':
        return <Bug className="h-4 w-4" />;
      case 'date':
        return <Calendar className="h-4 w-4" />;
      default:
        return <FileText className="h-4 w-4" />;
    }
  };

  const getEntityColor = (type: string) => {
    switch (type) {
      case 'crop':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'location':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'pest':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'date':
        return 'bg-purple-100 text-purple-800 border-purple-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const highlightEntities = (text: string, entities: Entity[]) => {
    if (entities.length === 0) return text;

    const sortedEntities = [...entities].sort((a, b) => b.start - a.start);
    let highlightedText = text;

    sortedEntities.forEach(entity => {
      const before = highlightedText.substring(0, entity.start);
      const entityText = highlightedText.substring(entity.start, entity.end);
      const after = highlightedText.substring(entity.end);
      
      highlightedText = before + 
        `<span class="px-2 py-1 rounded border-2 ${getEntityColor(entity.type)} font-semibold">` +
        entityText + 
        '</span>' + 
        after;
    });

    return highlightedText;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-green-800 mb-2">{t.title}</h1>
        <p className="text-green-600">{t.subtitle}</p>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="space-y-4">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={t.placeholder}
            className="w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 resize-none"
          />
          
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={analyzeText}
              disabled={!inputText.trim() || isAnalyzing}
              className="flex items-center justify-center space-x-2 px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isAnalyzing ? (
                <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
              ) : (
                <Search className="h-5 w-5" />
              )}
              <span>{t.analyze}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Sample Texts */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t.examples}</h3>
        <div className="grid gap-3">
          {sampleTexts[language as keyof typeof sampleTexts].map((sample, index) => (
            <button
              key={index}
              onClick={() => setInputText(sample)}
              className="text-left p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors border border-gray-200"
            >
              <p className="text-sm text-gray-700">{sample}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Results */}
      {entities.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
          <h2 className="text-2xl font-bold text-gray-800">{t.results}</h2>
          
          {/* Highlighted Text */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Highlighted Text</h3>
            <div 
              className="text-gray-700 leading-relaxed"
              dangerouslySetInnerHTML={{ __html: highlightEntities(inputText, entities) }}
            />
          </div>

          {/* Entity List */}
          <div className="grid gap-3">
            <h3 className="text-lg font-semibold text-gray-800">Extracted Entities</h3>
            {entities.map((entity, index) => (
              <div key={index} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${getEntityColor(entity.type)}`}>
                    {getEntityIcon(entity.type)}
                  </div>
                  <div>
                    <p className="font-semibold text-gray-800">{entity.text}</p>
                    <p className="text-sm text-gray-600">
                      {t[entity.type as keyof typeof t] || entity.type}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-gray-700">
                    {Math.round(entity.confidence * 100)}%
                  </p>
                  <p className="text-xs text-gray-500">{t.confidence}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {entities.length === 0 && inputText && !isAnalyzing && (
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">{t.noEntities}</p>
        </div>
      )}
    </div>
  );
};