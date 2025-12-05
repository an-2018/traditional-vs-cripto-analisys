# Comparação Gestão de Carteiras Tradicionais e Criptomoedas

## Visão Geral
Este projeto tem como objetivo realizar uma análise quantitativa comparativa entre a gestão de carteiras tradicionais e carteiras que incluem criptomoedas. O estudo foca em aspectos de risco, retorno e diversificação, utilizando dados históricos dos últimos 5 a 10 anos e projetando cenários futuros.

## Objetivos
- **Análise Histórica**: Comparar desempenho, volatilidade, Sharpe Ratio e Drawdowns.
- **Diversificação**: Avaliar o impacto da inclusão de criptoativos na correlação e eficiência do portfólio.
- **Projeções**: Utilizar modelos estatísticos (ARIMA, GARCH) e ML para prever cenários futuros.

## Estrutura do Projeto
```
source/
├── data/               # Dados brutos e processados
├── notebooks/          # Notebooks Jupyter para análise
├── src/                # Código fonte (ingestão, processamento, modelos)
├── pyproject.toml      # Gerenciamento de dependências (uv)
└── README.md
```

## Configuração do Ambiente
Este projeto utiliza `uv` para gerenciamento de dependências.

1.  **Instalar uv** (se não tiver):
    ```bash
    pip install uv
    ```

2.  **Instalar dependências**:
    ```bash
    uv sync
    ```

3.  **Executar Notebooks**:
    ```bash
    uv run jupyter notebook
    ```

## Metodologia
1.  **Coleta de Dados**: `yfinance` (ações) e `ccxt` (cripto).
2.  **Processamento**: Limpeza e alinhamento de séries temporais.
3.  **Análise**: Cálculo de métricas financeiras e otimização de portfólio (Markowitz).
4.  **Previsão**: Modelagem de séries temporais e volatilidade.

## Como Usar

### 1. Executar o Pipeline de Dados

```bash
# Executar pipeline standalone (6 meses)
uv run python src/run_pipeline.py
```

Ou usar o workflow Prefect (com visualização):

```bash
# Executar workflow Prefect
uv run python src/workflows/data_pipeline.py

# Iniciar servidor Prefect (opcional, para UI)
uv run prefect server start
```

### 2. Análise no Jupyter Notebook

```bash
# Iniciar Jupyter
uv run jupyter notebook

# Abrir: notebooks/01_portfolio_analysis_6months.ipynb
```

### 3. Estrutura dos Dados

- **Raw data**: `data/raw/` - Dados brutos por fonte
- **Processed data**: `data/processed/` - Dados limpos e prontos para análise

## Próximos Passos

- [ ] Expandir análise para 1, 2, 5 e 10 anos
- [ ] Implementar otimização de portfólio (Markowitz, Black-Litterman)
- [ ] Adicionar modelos preditivos (ARIMA, GARCH, ML)
- [ ] Criar dashboard interativo

