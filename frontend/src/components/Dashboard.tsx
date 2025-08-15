import React from 'react';
import { 
  TrendingUp, 
  AlertTriangle, 
  Cloud, 
  DollarSign,
  Thermometer,
  Droplets,
  Wind,
  Sun
} from 'lucide-react';

interface DashboardProps {
  language: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ language }) => {
  const translations = {
    en: {
      title: 'Agricultural Dashboard',
      subtitle: 'Real-time insights for better farming decisions',
      weather: 'Weather Forecast',
      alerts: 'Recent Alerts',
      finance: 'Financial Overview',
      crops: 'Crop Status',
      temperature: 'Temperature',
      humidity: 'Humidity',
      windSpeed: 'Wind Speed',
      rainfall: 'Rainfall',
      diseaseAlert: 'Leaf blight detected in tomato crop',
      weatherAlert: 'Heavy rainfall expected next week',
      priceAlert: 'Wheat prices increased by 8%',
      totalIncome: 'Total Income',
      expenses: 'Expenses',
      profit: 'Net Profit',
      healthy: 'Healthy',
      atRisk: 'At Risk',
      infected: 'Infected'
    },
    hi: {
      title: 'कृषि डैशबोर्ड',
      subtitle: 'बेहतर खेती के निर्णयों के लिए वास्तविक समय की जानकारी',
      weather: 'मौसम पूर्वानुमान',
      alerts: 'हाल की चेतावनियां',
      finance: 'वित्तीय अवलोकन',
      crops: 'फसल स्थिति',
      temperature: 'तापमान',
      humidity: 'नमी',
      windSpeed: 'हवा की गति',
      rainfall: 'वर्षा',
      diseaseAlert: 'टमाटर की फसल में पत्ती झुलसा रोग का पता चला',
      weatherAlert: 'अगले सप्ताह भारी बारिश की संभावना',
      priceAlert: 'गेहूं की कीमतों में 8% की वृद्धि',
      totalIncome: 'कुल आय',
      expenses: 'खर्च',
      profit: 'शुद्ध लाभ',
      healthy: 'स्वस्थ',
      atRisk: 'जोखिम में',
      infected: 'संक्रमित'
    }
  };

  const t = translations[language as keyof typeof translations];

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-green-800 mb-2">{t.title}</h1>
        <p className="text-green-600">{t.subtitle}</p>
      </div>

      {/* Weather Section */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <Cloud className="h-5 w-5 mr-2 text-blue-500" />
          {t.weather}
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-orange-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{t.temperature}</p>
                <p className="text-2xl font-bold text-orange-600">28°C</p>
              </div>
              <Thermometer className="h-8 w-8 text-orange-500" />
            </div>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{t.humidity}</p>
                <p className="text-2xl font-bold text-blue-600">65%</p>
              </div>
              <Droplets className="h-8 w-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{t.windSpeed}</p>
                <p className="text-2xl font-bold text-gray-600">12 km/h</p>
              </div>
              <Wind className="h-8 w-8 text-gray-500" />
            </div>
          </div>
          <div className="bg-yellow-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">{t.rainfall}</p>
                <p className="text-2xl font-bold text-yellow-600">15mm</p>
              </div>
              <Sun className="h-8 w-8 text-yellow-500" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Alerts Section */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-red-500" />
            {t.alerts}
          </h2>
          <div className="space-y-3">
            <div className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-red-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800">{t.diseaseAlert}</p>
                <p className="text-xs text-red-600">2 hours ago</p>
              </div>
            </div>
            <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
              <Cloud className="h-5 w-5 text-yellow-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-yellow-800">{t.weatherAlert}</p>
                <p className="text-xs text-yellow-600">5 hours ago</p>
              </div>
            </div>
            <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
              <TrendingUp className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-green-800">{t.priceAlert}</p>
                <p className="text-xs text-green-600">1 day ago</p>
              </div>
            </div>
          </div>
        </div>

        {/* Financial Overview */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
            <DollarSign className="h-5 w-5 mr-2 text-green-500" />
            {t.finance}
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
              <span className="text-green-700 font-medium">{t.totalIncome}</span>
              <span className="text-2xl font-bold text-green-600">₹1,25,000</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
              <span className="text-red-700 font-medium">{t.expenses}</span>
              <span className="text-2xl font-bold text-red-600">₹85,000</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
              <span className="text-blue-700 font-medium">{t.profit}</span>
              <span className="text-2xl font-bold text-blue-600">₹40,000</span>
            </div>
          </div>
        </div>
      </div>

      {/* Crop Status */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">{t.crops}</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl font-bold text-white">75%</span>
            </div>
            <h3 className="font-semibold text-green-800">{t.healthy}</h3>
            <p className="text-sm text-green-600">Tomato, Wheat, Rice</p>
          </div>
          <div className="text-center p-4 bg-yellow-50 rounded-lg">
            <div className="w-16 h-16 bg-yellow-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl font-bold text-white">20%</span>
            </div>
            <h3 className="font-semibold text-yellow-800">{t.atRisk}</h3>
            <p className="text-sm text-yellow-600">Cotton, Sugarcane</p>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-2xl font-bold text-white">5%</span>
            </div>
            <h3 className="font-semibold text-red-800">{t.infected}</h3>
            <p className="text-sm text-red-600">Potato</p>
          </div>
        </div>
      </div>
    </div>
  );
};