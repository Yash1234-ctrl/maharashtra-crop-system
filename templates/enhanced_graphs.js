/**
 * Enhanced Graph Rendering System for Maharashtra Krushi Mitra
 * Improved weather and soil analysis visualizations
 */

// Enhanced Weather Graphs
function loadEnhancedWeatherGraphs(district) {
    const weatherGraphContainer = document.getElementById('weather-graphs-container');
    if (!weatherGraphContainer) return;

    // Show loading state
    weatherGraphContainer.innerHTML = `
        <div class="flex items-center justify-center h-64">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
            <span class="ml-3 text-gray-300">Loading enhanced weather trends...</span>
        </div>
    `;

    fetch(`/api/enhanced_weather_graphs/${district}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                renderWeatherGraphs(data.graphs, data.weather_summary);
            } else {
                showGraphError('Failed to load weather graphs');
            }
        })
        .catch(error => {
            console.error('Weather graphs error:', error);
            showGraphError('Error loading weather data');
        });
}

function renderWeatherGraphs(graphs, summary) {
    const container = document.getElementById('weather-graphs-container');
    if (!container) return;

    container.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Temperature Graph -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="temperature-graph" class="trend-graph"></div>
            </div>
            
            <!-- Humidity Graph -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="humidity-graph" class="trend-graph"></div>
            </div>
            
            <!-- Rainfall Graph -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="rainfall-graph" class="trend-graph"></div>
            </div>
            
            <!-- Wind Speed Graph -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="wind-graph" class="trend-graph"></div>
            </div>
        </div>
        
        <!-- Weather Summary -->
        <div class="mt-6 glass-morphism rounded-xl p-6">
            <h3 class="text-lg font-semibold text-green-400 mb-4">📊 Weather Impact Analysis</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">Temperature Impact</h4>
                    <p class="text-lg font-medium ${summary.temperature_impact === 'Optimal' ? 'text-green-400' : 'text-yellow-400'}">
                        ${summary.temperature_impact}
                    </p>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">Humidity Impact</h4>
                    <p class="text-lg font-medium ${summary.humidity_impact === 'Good' ? 'text-green-400' : 'text-yellow-400'}">
                        ${summary.humidity_impact}
                    </p>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">Rainfall Impact</h4>
                    <p class="text-lg font-medium ${summary.rainfall_impact === 'Adequate' ? 'text-green-400' : 'text-yellow-400'}">
                        ${summary.rainfall_impact}
                    </p>
                </div>
            </div>
        </div>
    `;

    // Render Plotly graphs
    Plotly.newPlot('temperature-graph', JSON.parse(graphs.temperature_graph), {}, {responsive: true});
    Plotly.newPlot('humidity-graph', JSON.parse(graphs.humidity_graph), {}, {responsive: true});
    Plotly.newPlot('rainfall-graph', JSON.parse(graphs.rainfall_graph), {}, {responsive: true});
    Plotly.newPlot('wind-graph', JSON.parse(graphs.wind_graph), {}, {responsive: true});
}

// Enhanced Soil Graphs
function loadEnhancedSoilGraphs(soilData, district) {
    const soilGraphContainer = document.getElementById('soil-graphs-container');
    if (!soilGraphContainer) return;

    // Show loading state
    soilGraphContainer.innerHTML = `
        <div class="flex items-center justify-center h-64">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
            <span class="ml-3 text-gray-300">Loading enhanced soil analysis...</span>
        </div>
    `;

    // Prepare form data
    const formData = new FormData();
    formData.append('nitrogen', soilData.nitrogen || 50);
    formData.append('phosphorus', soilData.phosphorus || 30);
    formData.append('potassium', soilData.potassium || 40);
    formData.append('ph', soilData.ph || 6.5);
    formData.append('district', district || 'Pune');

    fetch('/api/enhanced_soil_graphs', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            renderSoilGraphs(data.graphs, data.soil_analysis);
        } else {
            showGraphError('Failed to load soil graphs');
        }
    })
    .catch(error => {
        console.error('Soil graphs error:', error);
        showGraphError('Error loading soil data');
    });
}

function renderSoilGraphs(graphs, analysis) {
    const container = document.getElementById('soil-graphs-container');
    if (!container) return;

    container.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- NPK Levels Chart -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="npk-graph" class="health-graph"></div>
            </div>
            
            <!-- Soil Health Gauge -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="health-gauge" class="health-graph"></div>
            </div>
            
            <!-- pH Level Gauge -->
            <div class="glass-morphism rounded-xl p-4">
                <div id="ph-gauge" class="health-graph"></div>
            </div>
            
            <!-- Health Trends (if available) -->
            ${graphs.trend_graph ? `
                <div class="glass-morphism rounded-xl p-4">
                    <div id="trend-graph" class="health-graph"></div>
                </div>
            ` : `
                <div class="glass-morphism rounded-xl p-4 flex items-center justify-center">
                    <div class="text-center">
                        <div class="text-gray-400 text-4xl mb-2">📈</div>
                        <p class="text-gray-400">Trend data available after multiple analyses</p>
                    </div>
                </div>
            `}
        </div>
        
        <!-- Soil Analysis Summary -->
        <div class="mt-6 glass-morphism rounded-xl p-6">
            <h3 class="text-lg font-semibold text-green-400 mb-4">🧪 Soil Analysis Summary</h3>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">Overall Health</h4>
                    <p class="text-2xl font-bold ${getHealthColor(analysis.soil_score)}">${analysis.soil_score}/100</p>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">pH Level</h4>
                    <p class="text-xl font-medium text-blue-400">${analysis.ph}</p>
                    <span class="text-xs ${getStatusColor(analysis.ph_status)}">${analysis.ph_status}</span>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">NPK Balance</h4>
                    <div class="flex space-x-2 mt-1">
                        <span class="px-2 py-1 text-xs rounded ${getStatusColor(analysis.npk_balance.N.status)}">
                            N: ${analysis.npk_balance.N.status}
                        </span>
                        <span class="px-2 py-1 text-xs rounded ${getStatusColor(analysis.npk_balance.P.status)}">
                            P: ${analysis.npk_balance.P.status}
                        </span>
                        <span class="px-2 py-1 text-xs rounded ${getStatusColor(analysis.npk_balance.K.status)}">
                            K: ${analysis.npk_balance.K.status}
                        </span>
                    </div>
                </div>
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="text-sm text-gray-400">Recommendations</h4>
                    <p class="text-sm text-gray-300">${analysis.recommendations.length} suggestions</p>
                </div>
            </div>
        </div>
    `;

    // Render Plotly graphs
    Plotly.newPlot('npk-graph', JSON.parse(graphs.npk_graph), {}, {responsive: true});
    Plotly.newPlot('health-gauge', JSON.parse(graphs.health_gauge), {}, {responsive: true});
    Plotly.newPlot('ph-gauge', JSON.parse(graphs.ph_gauge), {}, {responsive: true});
    
    if (graphs.trend_graph) {
        Plotly.newPlot('trend-graph', JSON.parse(graphs.trend_graph), {}, {responsive: true});
    }
}

// Comprehensive Dashboard Graphs
function loadComprehensiveGraphs(district, soilData) {
    const comprehensiveContainer = document.getElementById('comprehensive-graphs-container');
    if (!comprehensiveContainer) return;

    // Show loading state
    comprehensiveContainer.innerHTML = `
        <div class="flex items-center justify-center h-64">
            <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-green-500"></div>
            <span class="ml-4 text-gray-300 text-lg">Loading comprehensive analysis...</span>
        </div>
    `;

    // Prepare form data
    const formData = new FormData();
    formData.append('nitrogen', soilData.nitrogen || 50);
    formData.append('phosphorus', soilData.phosphorus || 30);
    formData.append('potassium', soilData.potassium || 40);
    formData.append('ph', soilData.ph || 6.5);

    fetch(`/api/comprehensive_graphs/${district}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            renderComprehensiveGraphs(data);
        } else {
            showGraphError('Failed to load comprehensive graphs');
        }
    })
    .catch(error => {
        console.error('Comprehensive graphs error:', error);
        showGraphError('Error loading comprehensive data');
    });
}

function renderComprehensiveGraphs(data) {
    const container = document.getElementById('comprehensive-graphs-container');
    if (!container) return;

    container.innerHTML = `
        <div class="space-y-8">
            <!-- Weather Section -->
            <div class="bg-gradient-to-r from-blue-900/20 to-blue-800/20 rounded-xl p-6 border border-blue-500/20">
                <h2 class="text-xl font-bold text-blue-400 mb-4">🌤️ Weather Analysis Dashboard</h2>
                <div id="weather-section"></div>
            </div>
            
            <!-- Soil Section -->
            <div class="bg-gradient-to-r from-green-900/20 to-green-800/20 rounded-xl p-6 border border-green-500/20">
                <h2 class="text-xl font-bold text-green-400 mb-4">🌱 Soil Health Dashboard</h2>
                <div id="soil-section"></div>
            </div>
        </div>
    `;

    // Render weather graphs
    const weatherSection = document.getElementById('weather-section');
    weatherSection.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-temperature-graph" class="trend-graph"></div>
            </div>
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-humidity-graph" class="trend-graph"></div>
            </div>
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-rainfall-graph" class="trend-graph"></div>
            </div>
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-wind-graph" class="trend-graph"></div>
            </div>
        </div>
    `;

    // Render soil graphs
    const soilSection = document.getElementById('soil-section');
    soilSection.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-npk-graph" class="health-graph"></div>
            </div>
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-health-gauge" class="health-graph"></div>
            </div>
            <div class="glass-morphism rounded-lg p-3">
                <div id="comp-ph-gauge" class="health-graph"></div>
            </div>
            ${data.soil_graphs.trend_graph ? `
                <div class="glass-morphism rounded-lg p-3">
                    <div id="comp-trend-graph" class="health-graph"></div>
                </div>
            ` : `
                <div class="glass-morphism rounded-lg p-3 flex items-center justify-center">
                    <div class="text-center text-gray-400">
                        <div class="text-3xl mb-2">📊</div>
                        <p class="text-sm">Trend analysis available after multiple tests</p>
                    </div>
                </div>
            `}
        </div>
    `;

    // Plot all weather graphs
    if (data.weather_graphs) {
        Plotly.newPlot('comp-temperature-graph', JSON.parse(data.weather_graphs.temperature_graph), {}, {responsive: true});
        Plotly.newPlot('comp-humidity-graph', JSON.parse(data.weather_graphs.humidity_graph), {}, {responsive: true});
        Plotly.newPlot('comp-rainfall-graph', JSON.parse(data.weather_graphs.rainfall_graph), {}, {responsive: true});
        Plotly.newPlot('comp-wind-graph', JSON.parse(data.weather_graphs.wind_graph), {}, {responsive: true});
    }

    // Plot all soil graphs
    if (data.soil_graphs) {
        Plotly.newPlot('comp-npk-graph', JSON.parse(data.soil_graphs.npk_graph), {}, {responsive: true});
        Plotly.newPlot('comp-health-gauge', JSON.parse(data.soil_graphs.health_gauge), {}, {responsive: true});
        Plotly.newPlot('comp-ph-gauge', JSON.parse(data.soil_graphs.ph_gauge), {}, {responsive: true});
        
        if (data.soil_graphs.trend_graph) {
            Plotly.newPlot('comp-trend-graph', JSON.parse(data.soil_graphs.trend_graph), {}, {responsive: true});
        }
    }
}

// Utility Functions
function getHealthColor(score) {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
}

function getStatusColor(status) {
    switch (status.toLowerCase()) {
        case 'optimal':
        case 'good':
        case 'neutral':
            return 'bg-green-600 text-white';
        case 'medium':
        case 'moderate':
            return 'bg-yellow-600 text-white';
        case 'low':
        case 'acidic':
        case 'alkaline':
            return 'bg-red-600 text-white';
        default:
            return 'bg-gray-600 text-white';
    }
}

function showGraphError(message) {
    const containers = ['weather-graphs-container', 'soil-graphs-container', 'comprehensive-graphs-container'];
    
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="flex items-center justify-center h-64 glass-morphism rounded-xl">
                    <div class="text-center">
                        <div class="text-red-400 text-4xl mb-4">⚠️</div>
                        <p class="text-red-400 text-lg">${message}</p>
                        <button onclick="location.reload()" class="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition">
                            Retry
                        </button>
                    </div>
                </div>
            `;
        }
    });
}

// Auto-refresh graphs every 5 minutes for real-time data
function enableAutoRefresh(district, soilData) {
    setInterval(() => {
        if (document.getElementById('weather-graphs-container')) {
            loadEnhancedWeatherGraphs(district);
        }
        
        if (document.getElementById('soil-graphs-container')) {
            loadEnhancedSoilGraphs(soilData, district);
        }
        
        if (document.getElementById('comprehensive-graphs-container')) {
            loadComprehensiveGraphs(district, soilData);
        }
    }, 300000); // 5 minutes
}

// Initialize enhanced graphs when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on a page that needs enhanced graphs
    const district = document.getElementById('districtSelect')?.value || 'Pune';
    
    // Default soil data for demo
    const defaultSoilData = {
        nitrogen: 50,
        phosphorus: 30,
        potassium: 40,
        ph: 6.5
    };
    
    // Load graphs based on available containers
    if (document.getElementById('weather-graphs-container')) {
        loadEnhancedWeatherGraphs(district);
    }
    
    if (document.getElementById('soil-graphs-container')) {
        loadEnhancedSoilGraphs(defaultSoilData, district);
    }
    
    if (document.getElementById('comprehensive-graphs-container')) {
        loadComprehensiveGraphs(district, defaultSoilData);
    }
    
    // Enable auto-refresh if needed
    // enableAutoRefresh(district, defaultSoilData);
});