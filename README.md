# KrypOracle - Ferramenta de Análise de Mercado de Criptomoedas

![KrypOracle Logo](data/assets/icon.png)

KrypOracle é uma poderosa ferramenta Python para análise de mercados de criptomoedas, projetada para ajudar traders e investidores a tomar decisões mais informadas com base em dados históricos e indicadores técnicos.

## Características Principais

- **Coleta de Dados**: Extração automática de dados OHLCV de múltiplas exchanges (Binance, Coinbase)
- **Análise Técnica**: Ampla biblioteca de indicadores técnicos:
  - **Tendências**: SMA, EMA, MACD, RSI, Golden/Death Cross
  - **Volatilidade**: Bollinger Bands, ATR, Keltner Channels
- **Visualização Avançada**: Gráficos interativos e dashboards para análise visual
- **Backtesting**: Sistema completo para testar estratégias de trading em dados históricos
- **Processamento de Dados**: Ferramentas para limpeza, normalização e transformação de dados

## Instalação

### Requisitos Prévios

- Python 3.8+
- Poetry (opcional, mas recomendado para gerenciamento de dependências)

### Via Poetry (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/raulneto-3/kryporacle.git
cd kryporacle

# Instale as dependências
poetry install
```

### Via Pip

```bash
# Clone o repositório
git clone https://github.com/raulneto-3/kryporacle.git
cd kryporacle

# Instale as dependências
pip install -r requirements.txt
```

## Configuração

1. Copie o arquivo de exemplo de configuração:
```bash
cp .env.example .env
```

2. Edite o arquivo `.env` com suas chaves de API:
```
# Exchange API Keys
BINANCE_API_KEY=sua_chave_api_binance
BINANCE_API_SECRET=seu_segredo_api_binance
COINBASE_API_KEY=sua_chave_api_coinbase
COINBASE_API_SECRET=seu_segredo_api_coinbase

# Storage Configuration
DATA_STORAGE_PATH=./data/storage
BACKTESTS_PATH=./data/backtests
```

## Exemplos de Uso

### Coleta de Dados Históricos

```python
from kryporacle.data.exchanges.binance import BinanceCollector
from kryporacle.data.storage import DataStorage
from datetime import datetime, timedelta

# Inicializar componentes
collector = BinanceCollector()
storage = DataStorage()

# Definir período para coleta de dados
end_time = datetime.now()
start_time = end_time - timedelta(days=30)

# Coletar dados históricos
data = collector.fetch_historical_data(
    symbol="BTC/USDT",
    timeframe="4h",
    start_time=start_time,
    end_time=end_time
)

# Salvar dados
storage.save_market_data("binance", data)
print(f"Coletados e salvos {len(data['data'])} registros")
```

### Análise Técnica

```python
from kryporacle.data.storage import DataStorage
from kryporacle.processing.processor import DataProcessor
from kryporacle.analysis import trends, volatility

# Inicializar componentes
storage = DataStorage()
processor = DataProcessor(storage)

# Carregar e processar dados
df = processor.process_market_data("binance", "BTC/USDT", "4h")

# Calcular indicadores técnicos
df = trends.calculate_indicators(df)  # SMA, EMA, MACD, RSI
df = volatility.calculate_volatility_indicators(df)  # Bollinger Bands, ATR

print(f"Dados processados com {df.shape[0]} registros e {df.shape[1]} indicadores")
```

### Visualização

```python
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.indicators import IndicatorVisualizer
from kryporacle.visualization.dashboard import Dashboard

# Criar gráfico de candlestick
ohlc_chart = OHLCChart()
fig, ax = ohlc_chart.plot_candlestick(df, title="BTC/USDT")
ohlc_chart.save_figure(fig, "candlestick.png")

# Visualizar indicadores
indicator_viz = IndicatorVisualizer()
fig, ax = indicator_viz.plot_bollinger_bands(df)
indicator_viz.save_figure(fig, "bollinger_bands.png")

# Criar dashboard completo
dashboard = Dashboard()
fig, axes = dashboard.create_market_dashboard(df, title="Análise de Mercado BTC/USDT")
dashboard.save_figure(fig, "dashboard.png")
```

### Backtesting

```python
from kryporacle.backtesting.engine import BacktestEngine
from kryporacle.backtesting.strategies import MovingAverageCrossover
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.visualizer import plot_backtest_results

# Inicializar backtesting
engine = BacktestEngine(initial_capital=10000.0, commission=0.001)

# Criar estratégia
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)

# Executar backtest
results = engine.run(
    strategy=strategy,
    exchange_id="binance",
    symbol="BTC/USDT",
    timeframe="1d",
    start_time="2022-01-01",
    end_time="2022-12-31"
)

# Calcular métricas de performance
metrics = calculate_metrics(results)
print(f"Retorno Total: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Visualizar resultados
fig = plot_backtest_results(results, strategy.name)
fig.savefig("backtest_results.png")
```

## Estrutura do Projeto

```
kryporacle/
├── data/                  # Coleta e armazenamento de dados
│   ├── exchanges/         # Implementações para diferentes exchanges
│   └── storage.py         # Persistência e recuperação de dados
├── processing/            # Processamento e transformação de dados
│   ├── cleaning.py        # Funções de limpeza de dados
│   ├── normalization.py   # Normalização e padronização
│   ├── aggregation.py     # Funções de agregação
│   └── processor.py       # Classe principal de processamento
├── analysis/              # Análise técnica
│   ├── trends.py          # Indicadores de tendência
│   └── volatility.py      # Indicadores de volatilidade
├── visualization/         # Visualização de dados
│   ├── charts.py          # Gráficos básicos
│   ├── indicators.py      # Visualização de indicadores
│   └── dashboard.py       # Dashboards compostos
├── backtesting/           # Sistema de backtesting
│   ├── engine.py          # Motor de backtesting
│   ├── strategies.py      # Estratégias básicas
│   ├── advanced_strategies.py # Estratégias avançadas
│   ├── metrics.py         # Cálculo de métricas
│   └── visualizer.py      # Visualização de resultados
└── config/                # Configurações do sistema
    └── settings.py        # Configurações globais
```

## Estratégias de Trading Incluídas

- **Básicas**:
  - Cruzamento de Médias Móveis
  - Estratégia baseada em RSI
  - Estratégia de Bandas de Bollinger
  - Estratégia de MACD

- **Avançadas**:
  - Seguimento de Tendência
  - MACD com Confirmação Dupla
  - Breakout de Volatilidade

## Licença

Este projeto é licenciado sob os termos da licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para enviar um Pull Request.

1. Faça um fork do projeto
2. Crie sua branch de feature (`git checkout -b feature/amazing-feature`)
3. Commit suas mudanças (`git commit -m 'Add some amazing feature'`)
4. Push para a branch (`git push origin feature/amazing-feature`)
5. Abra um Pull Request

<!-- ## Contato

Seu Nome - [@seu_twitter](https://twitter.com/seu_twitter) - email@exemplo.com -->

Link do Projeto: [https://github.com/raulneto-3/kryporacle](https://github.com/raulneto-3/kryporacle)