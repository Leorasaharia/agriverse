import React from 'react';
import { 
  Home, 
  Camera, 
  FileText, 
  MessageCircle, 
  BookOpen, 
  Globe2,
  Leaf
} from 'lucide-react';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  language: string;
  setLanguage: (lang: string) => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  language,
  setLanguage
}) => {
  const translations = {
    en: {
      title: 'AgriVerse AI',
      dashboard: 'Dashboard',
      diseaseDetection: 'Disease Detection',
      nerAnalysis: 'Text Analysis',
      chatbot: 'AI Assistant',
      knowledge: 'Knowledge Base'
    },
    hi: {
      title: 'एग्रीवर्स AI',
      dashboard: 'डैशबोर्ड',
      diseaseDetection: 'रोग पहचान',
      nerAnalysis: 'पाठ विश्लेषण',
      chatbot: 'AI सहायक',
      knowledge: 'ज्ञान आधार'
    }
  };

  const t = translations[language as keyof typeof translations];

  const navItems = [
    { id: 'dashboard', icon: Home, label: t.dashboard },
    { id: 'disease-detection', icon: Camera, label: t.diseaseDetection },
    { id: 'ner-analysis', icon: FileText, label: t.nerAnalysis },
    { id: 'chatbot', icon: MessageCircle, label: t.chatbot },
    { id: 'knowledge', icon: BookOpen, label: t.knowledge }
  ];

  return (
    <nav className="bg-white shadow-lg border-b-4 border-green-500">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-green-500 rounded-lg">
              <Leaf className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-xl font-bold text-green-800">{t.title}</h1>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setLanguage(language === 'en' ? 'hi' : 'en')}
              className="flex items-center space-x-2 px-3 py-2 rounded-lg bg-green-100 hover:bg-green-200 transition-colors"
            >
              <Globe2 className="h-4 w-4" />
              <span className="text-sm font-medium">
                {language === 'en' ? 'हिंदी' : 'English'}
              </span>
            </button>
          </div>
        </div>

        <div className="flex space-x-1 pb-4 overflow-x-auto">
          {navItems.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                activeTab === id
                  ? 'bg-green-500 text-white shadow-md'
                  : 'bg-green-50 text-green-700 hover:bg-green-100'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span className="text-sm font-medium">{label}</span>
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};