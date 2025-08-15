import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Mic, MicOff } from 'lucide-react';

interface QAChatbotProps {
  language: string;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export const QAChatbot: React.FC<QAChatbotProps> = ({ language }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const translations = {
    en: {
      title: 'AgriVerse AI Assistant',
      subtitle: 'Ask questions about farming, crops, weather, and agricultural finance',
      placeholder: 'Ask me anything about agriculture...',
      send: 'Send',
      typing: 'AgriVerse AI is typing...',
      voiceInput: 'Voice Input',
      sampleQuestions: 'Sample Questions',
      questions: [
        'When should I irrigate my wheat crop?',
        'What seed variety suits unpredictable weather?',
        'Will next week\'s temperature drop kill my yield?',
        'Can I afford to wait for the market to improve?',
        'Where can I get affordable agricultural credit?',
        'What government policies can help with finances?'
      ],
      welcome: 'Hello! I\'m your AgriVerse AI assistant. I can help you with farming questions, crop management, weather advice, and agricultural finance. How can I assist you today?'
    },
    hi: {
      title: 'एग्रीवर्स AI सहायक',
      subtitle: 'खेती, फसलों, मौसम और कृषि वित्त के बारे में सवाल पूछें',
      placeholder: 'कृषि के बारे में कुछ भी पूछें...',
      send: 'भेजें',
      typing: 'एग्रीवर्स AI टाइप कर रहा है...',
      voiceInput: 'आवाज़ इनपुट',
      sampleQuestions: 'नमूना प्रश्न',
      questions: [
        'मुझे अपनी गेहूं की फसल को कब सिंचाई करनी चाहिए?',
        'अप्रत्याशित मौसम के लिए कौन सी बीज किस्म उपयुक्त है?',
        'क्या अगले सप्ताह तापमान में गिरावट से मेरी उपज मर जाएगी?',
        'क्या मैं बाज़ार के सुधरने का इंतज़ार कर सकता हूँ?',
        'मुझे किफायती कृषि ऋण कहाँ मिल सकता है?',
        'कौन सी सरकारी नीतियां वित्त में मदद कर सकती हैं?'
      ],
      welcome: 'नमस्कार! मैं आपका एग्रीवर्स AI सहायक हूँ। मैं आपको खेती के सवाल, फसल प्रबंधन, मौसम की सलाह, और कृषि वित्त में मदद कर सकता हूँ। आज मैं आपकी कैसे सहायता कर सकता हूँ?'
    }
  };

  const t = translations[language as keyof typeof translations];

  useEffect(() => {
    // Add welcome message on first load
    if (messages.length === 0) {
      setMessages([{
        id: '1',
        text: t.welcome,
        sender: 'bot',
        timestamp: new Date()
      }]);
    }
  }, [language]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const mockResponses = {
    en: {
      irrigation: "For wheat irrigation, monitor soil moisture at 6-8 inch depth. Water when moisture drops to 50-60% of field capacity. During grain filling stage, maintain adequate moisture. Generally, wheat needs 4-6 irrigations depending on rainfall and soil type.",
      seeds: "For unpredictable weather, choose drought-tolerant varieties like HD 2967, HD 3086 for wheat. For rice, consider Sahbhagi dhan or DRR dhan 42. These varieties can withstand both drought and excess water conditions.",
      temperature: "Wheat is sensitive to temperatures above 35°C during grain filling. If temperature drops below 10°C during flowering, it can reduce yield by 10-20%. Monitor weather forecasts and consider protective measures.",
      market: "Check current mandi prices and storage costs. If storage is affordable and prices are expected to rise by more than storage costs, waiting might be profitable. Consider government procurement prices as baseline.",
      credit: "You can get agricultural loans from banks, cooperative societies, and NBFCs. Interest rates range from 7-9% for priority sector lending. Check PM-KISAN scheme for income support and KCC for credit.",
      policies: "Key schemes include PM-KISAN (₹6000/year), Soil Health Card, crop insurance under PMFBY, and subsidized fertilizers. State governments also offer additional support schemes.",
      default: "I understand your agricultural concern. Could you provide more specific details about your crop, location, or farming situation so I can give you more targeted advice?"
    },
    hi: {
      irrigation: "गेहूं की सिंचाई के लिए, 6-8 इंच गहराई पर मिट्टी की नमी की निगरानी करें। जब नमी क्षेत्र क्षमता के 50-60% तक गिर जाए तो पानी दें। दाना भरने के चरण के दौरान, पर्याप्त नमी बनाए रखें।",
      seeds: "अप्रत्याशित मौसम के लिए, गेहूं के लिए HD 2967, HD 3086 जैसी सूखा सहनशील किस्मों का चुनाव करें। धान के लिए सहभागी धान या DRR धान 42 का विकल्प करें।",
      temperature: "गेहूं दाना भरने के दौरान 35°C से अधिक तापमान के लिए संवेदनशील है। फूल आने के दौरान तापमान 10°C से नीचे गिरने पर उपज में 10-20% की कमी हो सकती है।",
      market: "वर्तमान मंडी भावों और भंडारण लागत की जांच करें। यदि भंडारण किफायती है और कीमतों में भंडारण लागत से अधिक वृद्धि की उम्मीद है, तो इंतज़ार फायदेमंद हो सकता है।",
      credit: "आप बैंकों, सहकारी समितियों और NBFCs से कृषि ऋण ले सकते हैं। प्राथमिकता क्षेत्र ऋण के लिए ब्याज दरें 7-9% तक हैं। आय सहायता के लिए PM-KISAN योजना देखें।",
      policies: "मुख्य योजनाओं में PM-KISAN (₹6000/वर्ष), मृदा स्वास्थ्य कार्ड, PMFBY के तहत फसल बीमा, और सब्सिडी युक्त उर्वरक शामिल हैं।",
      default: "मैं आपकी कृषि संबंधी चिंता समझता हूँ। कृपया अपनी फसल, स्थान, या खेती की स्थिति के बारे में अधिक विशिष्ट विवरण प्रदान करें ताकि मैं आपको अधिक लक्षित सलाह दे सकूं।"
    }
  };

  const getResponse = (message: string): string => {
    const responses = mockResponses[language as keyof typeof mockResponses];
    const lowerMessage = message.toLowerCase();

    if (lowerMessage.includes('irrigat') || lowerMessage.includes('water') || lowerMessage.includes('सिंचाई')) {
      return responses.irrigation;
    }
    if (lowerMessage.includes('seed') || lowerMessage.includes('variety') || lowerMessage.includes('बीज') || lowerMessage.includes('किस्म')) {
      return responses.seeds;
    }
    if (lowerMessage.includes('temperature') || lowerMessage.includes('weather') || lowerMessage.includes('तापमान') || lowerMessage.includes('मौसम')) {
      return responses.temperature;
    }
    if (lowerMessage.includes('market') || lowerMessage.includes('price') || lowerMessage.includes('बाज़ार') || lowerMessage.includes('कीमत')) {
      return responses.market;
    }
    if (lowerMessage.includes('credit') || lowerMessage.includes('loan') || lowerMessage.includes('ऋण')) {
      return responses.credit;
    }
    if (lowerMessage.includes('policy') || lowerMessage.includes('scheme') || lowerMessage.includes('नीति') || lowerMessage.includes('योजना')) {
      return responses.policies;
    }
    
    return responses.default;
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Simulate AI response delay
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: getResponse(inputText),
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    }, 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startVoiceRecognition = () => {
    setIsListening(true);
    // Mock voice recognition - in real implementation, use Web Speech API
    setTimeout(() => {
      setIsListening(false);
      setInputText("When should I irrigate my crops?");
    }, 3000);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-green-800 mb-2">{t.title}</h1>
        <p className="text-green-600">{t.subtitle}</p>
      </div>

      {/* Sample Questions */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{t.sampleQuestions}</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {t.questions.map((question, index) => (
            <button
              key={index}
              onClick={() => setInputText(question)}
              className="text-left p-3 bg-green-50 rounded-lg hover:bg-green-100 transition-colors border border-green-200 text-sm"
            >
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Chat Messages */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex items-start space-x-3 max-w-xs lg:max-w-md ${
                message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}>
                <div className={`p-2 rounded-full ${
                  message.sender === 'user' ? 'bg-green-500' : 'bg-blue-500'
                }`}>
                  {message.sender === 'user' ? (
                    <User className="h-4 w-4 text-white" />
                  ) : (
                    <Bot className="h-4 w-4 text-white" />
                  )}
                </div>
                <div className={`p-3 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                  <p className={`text-xs mt-1 ${
                    message.sender === 'user' ? 'text-green-100' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start">
              <div className="flex items-start space-x-3">
                <div className="p-2 bg-blue-500 rounded-full">
                  <Bot className="h-4 w-4 text-white" />
                </div>
                <div className="bg-gray-100 p-3 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{t.typing}</p>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Message Input */}
        <div className="border-t border-gray-200 p-4">
          <div className="flex space-x-2">
            <button
              onClick={startVoiceRecognition}
              className={`p-3 rounded-lg transition-colors ${
                isListening
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              title={t.voiceInput}
            >
              {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
            </button>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={t.placeholder}
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
            />
            <button
              onClick={sendMessage}
              disabled={!inputText.trim()}
              className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};