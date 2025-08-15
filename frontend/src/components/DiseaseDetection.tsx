import React, { useState, useRef } from 'react';
import { Camera, Upload, AlertCircle, CheckCircle, Info, X } from 'lucide-react';

interface DiseaseDetectionProps {
  language: string;
}

interface DetectionResult {
  disease: string;
  confidence: number;
  severity: string;
  recommendations: string[];
  treatment: string[];
}

export const DiseaseDetection: React.FC<DiseaseDetectionProps> = ({ language }) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<DetectionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const translations = {
    en: {
      title: 'Crop Disease Detection',
      subtitle: 'Upload or capture an image of your crop for AI-powered disease analysis',
      uploadImage: 'Upload Image',
      takePhoto: 'Take Photo',
      analyzing: 'Analyzing image...',
      results: 'Analysis Results',
      disease: 'Detected Disease',
      confidence: 'Confidence',
      severity: 'Severity',
      recommendations: 'Recommendations',
      treatment: 'Treatment Options',
      tryAnother: 'Analyze Another Image',
      dragDrop: 'Drag and drop an image here, or click to select',
      supportedFormats: 'Supported formats: JPG, PNG, WEBP'
    },
    hi: {
      title: 'फसल रोग पहचान',
      subtitle: 'AI-संचालित रोग विश्लेषण के लिए अपनी फसल की तस्वीर अपलोड करें या खींचें',
      uploadImage: 'तस्वीर अपलोड करें',
      takePhoto: 'फोटो लें',
      analyzing: 'तस्वीर का विश्लेषण हो रहा है...',
      results: 'विश्लेषण परिणाम',
      disease: 'पाया गया रोग',
      confidence: 'विश्वास',
      severity: 'गंभीरता',
      recommendations: 'सिफारिशें',
      treatment: 'उपचार विकल्प',
      tryAnother: 'दूसरी तस्वीर का विश्लेषण करें',
      dragDrop: 'यहाँ एक तस्वीर खींचें और छोड़ें, या चुनने के लिए क्लिक करें',
      supportedFormats: 'समर्थित प्रारूप: JPG, PNG, WEBP'
    }
  };

  const t = translations[language as keyof typeof translations];

  const mockResults: { [key: string]: DetectionResult } = {
    tomato: {
      disease: language === 'hi' ? 'टमाटर की पत्ती झुलसा' : 'Tomato Leaf Blight',
      confidence: 92.5,
      severity: language === 'hi' ? 'मध्यम' : 'Moderate',
      recommendations: language === 'hi' ? [
        'प्रभावित पत्तियों को हटा दें',
        'पौधों के बीच हवा का संचार बढ़ाएं',
        'पानी देते समय पत्तियों पर पानी न डालें',
        'कॉपर-आधारित fungicide का प्रयोग करें'
      ] : [
        'Remove affected leaves immediately',
        'Improve air circulation between plants',
        'Avoid overhead watering',
        'Apply copper-based fungicide'
      ],
      treatment: language === 'hi' ? [
        'नीम का तेल स्प्रे (सप्ताह में दो बार)',
        'बेकिंग सोडा मिश्रण (1 चम्मच प्रति लीटर)',
        'व्यावसायिक fungicide (Mancozeb)',
        'जैविक नियंत्रण एजेंट (Trichoderma)'
      ] : [
        'Neem oil spray (twice weekly)',
        'Baking soda solution (1 tsp per liter)',
        'Commercial fungicide (Mancozeb)',
        'Biological control agent (Trichoderma)'
      ]
    },
    rice: {
      disease: language === 'hi' ? 'धान का भूरा धब्बा रोग' : 'Rice Brown Spot',
      confidence: 87.3,
      severity: language === 'hi' ? 'गंभीर' : 'Severe',
      recommendations: language === 'hi' ? [
        'बीज उपचार करें',
        'संतुलित उर्वरक का प्रयोग',
        'खेत में जल निकासी सुधारें',
        'प्रतिरोधी किस्मों का चुनाव'
      ] : [
        'Treat seeds before planting',
        'Use balanced fertilization',
        'Improve field drainage',
        'Choose resistant varieties'
      ],
      treatment: language === 'hi' ? [
        'Propiconazole स्प्रे',
        'कार्बेन्डाज़िम उपचार',
        'Iprobenphos छिड़काव',
        'जैविक नीम का तेल'
      ] : [
        'Propiconazole spray',
        'Carbendazim treatment',
        'Iprobenphos application',
        'Organic neem oil treatment'
      ]
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
        simulateAnalysis();
      };
      reader.readAsDataURL(file);
    }
  };

  const simulateAnalysis = () => {
    setIsAnalyzing(true);
    setResults(null);
    
    // Simulate AI processing time
    setTimeout(() => {
      const randomResult = Math.random() > 0.5 ? mockResults.tomato : mockResults.rice;
      setResults(randomResult);
      setIsAnalyzing(false);
    }, 3000);
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setResults(null);
    setIsAnalyzing(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getSeverityColor = (severity: string) => {
    if (severity.toLowerCase().includes('severe') || severity.includes('गंभीर')) {
      return 'text-red-600 bg-red-100';
    }
    if (severity.toLowerCase().includes('moderate') || severity.includes('मध्यम')) {
      return 'text-yellow-600 bg-yellow-100';
    }
    return 'text-green-600 bg-green-100';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-green-800 mb-2">{t.title}</h1>
        <p className="text-green-600">{t.subtitle}</p>
      </div>

      {!selectedImage && !isAnalyzing && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div
            className="border-2 border-dashed border-green-300 rounded-xl p-8 text-center hover:border-green-400 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">{t.dragDrop}</h3>
            <p className="text-sm text-gray-500 mb-4">{t.supportedFormats}</p>
            
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="flex items-center justify-center space-x-2 px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
              >
                <Upload className="h-5 w-5" />
                <span>{t.uploadImage}</span>
              </button>
              <button className="flex items-center justify-center space-x-2 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                <Camera className="h-5 w-5" />
                <span>{t.takePhoto}</span>
              </button>
            </div>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </div>
      )}

      {selectedImage && (
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="relative">
            <img
              src={selectedImage}
              alt="Uploaded crop"
              className="w-full h-64 sm:h-80 object-cover"
            />
            <button
              onClick={resetAnalysis}
              className="absolute top-4 right-4 p-2 bg-white rounded-full shadow-lg hover:shadow-xl transition-shadow"
            >
              <X className="h-5 w-5 text-gray-600" />
            </button>
          </div>
        </div>
      )}

      {isAnalyzing && (
        <div className="bg-white rounded-xl shadow-lg p-8 text-center">
          <div className="animate-spin h-12 w-12 border-4 border-green-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <h3 className="text-lg font-semibold text-gray-700 mb-2">{t.analyzing}</h3>
          <p className="text-gray-500">AI is analyzing the crop image for disease detection...</p>
        </div>
      )}

      {results && (
        <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-800">{t.results}</h2>
            <button
              onClick={resetAnalysis}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
            >
              {t.tryAnother}
            </button>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-3 p-4 bg-red-50 rounded-lg">
                <AlertCircle className="h-8 w-8 text-red-500" />
                <div>
                  <h3 className="font-semibold text-gray-800">{t.disease}</h3>
                  <p className="text-lg font-bold text-red-600">{results.disease}</p>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 bg-blue-50 rounded-lg">
                <CheckCircle className="h-8 w-8 text-blue-500" />
                <div>
                  <h3 className="font-semibold text-gray-800">{t.confidence}</h3>
                  <p className="text-lg font-bold text-blue-600">{results.confidence}%</p>
                </div>
              </div>

              <div className="flex items-center space-x-3 p-4 rounded-lg">
                <Info className="h-8 w-8 text-yellow-500" />
                <div>
                  <h3 className="font-semibold text-gray-800">{t.severity}</h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getSeverityColor(results.severity)}`}>
                    {results.severity}
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-3">{t.recommendations}</h3>
                <ul className="space-y-2">
                  {results.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500 mt-1 flex-shrink-0" />
                      <span className="text-gray-700 text-sm">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-800 mb-3">{t.treatment}</h3>
                <ul className="space-y-2">
                  {results.treatment.map((treatment, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <Info className="h-4 w-4 text-blue-500 mt-1 flex-shrink-0" />
                      <span className="text-gray-700 text-sm">{treatment}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};