import React, { useState } from 'react';
import { 
  BookOpen, 
  Cloud, 
  Leaf, 
  Bug, 
  DollarSign, 
  Calendar,
  ChevronRight,
  Search,
  Download
} from 'lucide-react';

interface KnowledgeBaseProps {
  language: string;
}

interface Article {
  id: string;
  title: string;
  category: string;
  content: string;
  tags: string[];
  readTime: number;
}

export const KnowledgeBase: React.FC<KnowledgeBaseProps> = ({ language }) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);

  const translations = {
    en: {
      title: 'Agricultural Knowledge Base',
      subtitle: 'Comprehensive resource for modern farming practices',
      search: 'Search knowledge base...',
      categories: 'Categories',
      all: 'All Topics',
      weather: 'Weather & Climate',
      crops: 'Crop Management',
      pests: 'Pest Control',
      finance: 'Agricultural Finance',
      seasons: 'Seasonal Guide',
      readMore: 'Read More',
      backToList: 'Back to Articles',
      readTime: 'min read',
      download: 'Download Guide',
      relatedArticles: 'Related Articles'
    },
    hi: {
      title: 'कृषि ज्ञान आधार',
      subtitle: 'आधुनिक कृषि प्रथाओं के लिए व्यापक संसाधन',
      search: 'ज्ञान आधार में खोजें...',
      categories: 'श्रेणियां',
      all: 'सभी विषय',
      weather: 'मौसम और जलवायु',
      crops: 'फसल प्रबंधन',
      pests: 'कीट नियंत्रण',
      finance: 'कृषि वित्त',
      seasons: 'मौसमी गाइड',
      readMore: 'और पढ़ें',
      backToList: 'लेखों पर वापस जाएं',
      readTime: 'मिनट पढ़ें',
      download: 'गाइड डाउनलोड करें',
      relatedArticles: 'संबंधित लेख'
    }
  };

  const t = translations[language as keyof typeof translations];

  const categories = [
    { id: 'all', label: t.all, icon: BookOpen },
    { id: 'weather', label: t.weather, icon: Cloud },
    { id: 'crops', label: t.crops, icon: Leaf },
    { id: 'pests', label: t.pests, icon: Bug },
    { id: 'finance', label: t.finance, icon: DollarSign },
    { id: 'seasons', label: t.seasons, icon: Calendar }
  ];

  const articles: Article[] = [
    {
      id: '1',
      title: language === 'hi' ? 'गेहूं की खेती के लिए मिट्टी तैयारी' : 'Soil Preparation for Wheat Cultivation',
      category: 'crops',
      content: language === 'hi' 
        ? 'गेहूं की सफल खेती के लिए मिट्टी की तैयारी अत्यंत महत्वपूर्ण है। मिट्टी का pH 6.0-7.5 के बीच होना चाहिए। खेत की 2-3 बार जुताई करें और कंपोस्ट खाद मिलाएं।'
        : 'Proper soil preparation is crucial for successful wheat cultivation. The soil pH should be between 6.0-7.5. Plow the field 2-3 times and incorporate compost.',
      tags: ['wheat', 'soil', 'preparation'],
      readTime: 5
    },
    {
      id: '2',
      title: language === 'hi' ? 'मानसून पूर्वानुमान और फसल योजना' : 'Monsoon Forecasting and Crop Planning',
      category: 'weather',
      content: language === 'hi'
        ? 'मानसून की सटीक भविष्यवाणी से किसान अपनी फसल की योजना बेहतर तरीके से बना सकते हैं। IMD के आंकड़ों का उपयोग करें और स्थानीय मौसम केंद्रों से संपर्क रखें।'
        : 'Accurate monsoon prediction helps farmers plan their crops better. Use IMD data and maintain contact with local weather stations.',
      tags: ['monsoon', 'weather', 'planning'],
      readTime: 7
    },
    {
      id: '3',
      title: language === 'hi' ? 'टमाटर में रोग प्रबंधन' : 'Disease Management in Tomatoes',
      category: 'pests',
      content: language === 'hi'
        ? 'टमाटर में मुख्य रोगों में झुलसा रोग, मोज़ेक वायरस और फल सड़न शामिल हैं। समय पर छिड़काव और प्रतिरोधी किस्मों का चुनाव करें।'
        : 'Major diseases in tomatoes include blight, mosaic virus, and fruit rot. Timely spraying and choosing resistant varieties is essential.',
      tags: ['tomato', 'disease', 'management'],
      readTime: 6
    },
    {
      id: '4',
      title: language === 'hi' ? 'कृषि ऋण और सब्सिडी योजनाएं' : 'Agricultural Loans and Subsidy Schemes',
      category: 'finance',
      content: language === 'hi'
        ? 'किसानों के लिए विभिन्न ऋण योजनाएं उपलब्ध हैं जैसे KCC, PM-KISAN, और फसल बीमा। इन योजनाओं का लाभ उठाने के लिए आवश्यक दस्तावेज तैयार रखें।'
        : 'Various loan schemes are available for farmers like KCC, PM-KISAN, and crop insurance. Keep necessary documents ready to avail these schemes.',
      tags: ['finance', 'loans', 'schemes'],
      readTime: 8
    },
    {
      id: '5',
      title: language === 'hi' ? 'खरीफ फसलों की बुआई का समय' : 'Kharif Crop Sowing Time',
      category: 'seasons',
      content: language === 'hi'
        ? 'खरीफ फसलों की बुआई जून-जुलाई में की जाती है। मानसून की शुरुआत के साथ धान, मक्का, और कपास की बुआई का सही समय निर्धारित करें।'
        : 'Kharif crops are sown in June-July. Determine the right time for sowing rice, maize, and cotton with the onset of monsoon.',
      tags: ['kharif', 'sowing', 'timing'],
      readTime: 4
    }
  ];

  const filteredArticles = articles.filter(article => {
    const matchesCategory = selectedCategory === 'all' || article.category === selectedCategory;
    const matchesSearch = article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         article.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         article.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  if (selectedArticle) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <button
          onClick={() => setSelectedArticle(null)}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 transition-colors"
        >
          <ChevronRight className="h-4 w-4 rotate-180" />
          <span>{t.backToList}</span>
        </button>

        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="p-8">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span>{selectedArticle.readTime} {t.readTime}</span>
                <span>•</span>
                <span>{selectedArticle.category}</span>
              </div>
              <button className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
                <Download className="h-4 w-4" />
                <span>{t.download}</span>
              </button>
            </div>

            <h1 className="text-3xl font-bold text-gray-800 mb-4">{selectedArticle.title}</h1>
            
            <div className="flex flex-wrap gap-2 mb-6">
              {selectedArticle.tags.map(tag => (
                <span key={tag} className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                  #{tag}
                </span>
              ))}
            </div>

            <div className="prose prose-lg max-w-none text-gray-700 leading-relaxed">
              <p>{selectedArticle.content}</p>
              
              {/* Extended content for demo */}
              <div className="mt-6 space-y-4">
                <h2 className="text-xl font-semibold text-gray-800">
                  {language === 'hi' ? 'विस्तृत जानकारी' : 'Detailed Information'}
                </h2>
                <p>
                  {language === 'hi' 
                    ? 'यह विषय किसानों के लिए अत्यंत महत्वपूर्ण है और इसका सही ज्ञान फसल की गुणवत्ता और उपज दोनों को बेहतर बनाता है। आधुनिक तकनीकों का उपयोग करके परंपरागत ज्ञान को मिलाना सबसे प्रभावी दृष्टिकोण है।'
                    : 'This topic is extremely important for farmers and proper knowledge improves both crop quality and yield. Combining traditional knowledge with modern techniques is the most effective approach.'
                  }
                </p>
                
                <h3 className="text-lg font-semibold text-gray-800">
                  {language === 'hi' ? 'मुख्य बिंदु' : 'Key Points'}
                </h3>
                <ul className="list-disc pl-6 space-y-2">
                  <li>{language === 'hi' ? 'समय पर कार्यान्वयन अत्यावश्यक है' : 'Timely implementation is essential'}</li>
                  <li>{language === 'hi' ? 'गुणवत्तापूर्ण सामग्री का उपयोग करें' : 'Use quality materials'}</li>
                  <li>{language === 'hi' ? 'नियमित निगरानी रखें' : 'Maintain regular monitoring'}</li>
                  <li>{language === 'hi' ? 'विशेषज्ञों से सलाह लें' : 'Consult with experts'}</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-green-800 mb-2">{t.title}</h1>
        <p className="text-green-600">{t.subtitle}</p>
      </div>

      {/* Search Bar */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="relative">
          <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t.search}
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
          />
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Categories Sidebar */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl shadow-lg p-6 sticky top-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">{t.categories}</h2>
            <div className="space-y-2">
              {categories.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setSelectedCategory(id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    selectedCategory === id
                      ? 'bg-green-500 text-white'
                      : 'bg-green-50 text-green-700 hover:bg-green-100'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Articles Grid */}
        <div className="lg:col-span-3">
          <div className="grid md:grid-cols-2 gap-6">
            {filteredArticles.map(article => (
              <div key={article.id} className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
                      {categories.find(c => c.id === article.category)?.label}
                    </span>
                    <span className="text-sm text-gray-500">{article.readTime} {t.readTime}</span>
                  </div>
                  
                  <h3 className="text-lg font-semibold text-gray-800 mb-3 line-clamp-2">
                    {article.title}
                  </h3>
                  
                  <p className="text-gray-600 text-sm mb-4 line-clamp-3">
                    {article.content}
                  </p>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    {article.tags.slice(0, 3).map(tag => (
                      <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                        #{tag}
                      </span>
                    ))}
                  </div>
                  
                  <button
                    onClick={() => setSelectedArticle(article)}
                    className="flex items-center space-x-2 text-green-600 hover:text-green-700 transition-colors"
                  >
                    <span className="text-sm font-medium">{t.readMore}</span>
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {filteredArticles.length === 0 && (
            <div className="bg-white rounded-xl shadow-lg p-8 text-center">
              <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-700 mb-2">
                {language === 'hi' ? 'कोई लेख नहीं मिला' : 'No Articles Found'}
              </h3>
              <p className="text-gray-500">
                {language === 'hi' 
                  ? 'कृपया अपनी खोज को समायोजित करें या अन्य श्रेणी चुनें'
                  : 'Please adjust your search or select a different category'
                }
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};