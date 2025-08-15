import React, { useState } from 'react';
import { Navigation } from './components/Navigation';
import { DiseaseDetection } from './components/DiseaseDetection';
import { NERAnalysis } from './components/NERAnalysis';
import { QAChatbot } from './components/QAChatbot';
import { KnowledgeBase } from './components/KnowledgeBase';
import { Dashboard } from './components/Dashboard';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [language, setLanguage] = useState('en');

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard language={language} />;
      case 'disease-detection':
        return <DiseaseDetection language={language} />;
      case 'ner-analysis':
        return <NERAnalysis language={language} />;
      case 'chatbot':
        return <QAChatbot language={language} />;
      case 'knowledge':
        return <KnowledgeBase language={language} />;
      default:
        return <Dashboard language={language} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-50">
      <Navigation 
        activeTab={activeTab} 
        setActiveTab={setActiveTab}
        language={language}
        setLanguage={setLanguage}
      />
      <main className="container mx-auto px-4 py-8">
        {renderActiveComponent()}
      </main>
    </div>
  );
}

export default App;