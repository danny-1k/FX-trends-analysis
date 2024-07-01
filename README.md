This is an attempt to building a hybrid ML trading strategy.

Hybrid in the sense that we make use of rule-based, explainable algorithms and machine learning techniques.


## Hypothesis & Background:
It can be helpful to see the financial markets as a large interconnected system.
Changes in one asset may affect certain other assets in the market.

Oil prices can affect prices of certain currencies and in response, governments can change interest rates and so on and so forth.

Based on this analogy, we can notice how we can infer the prices or short term movements of constituents of the markets from observing the other assets in the market although with some lag.

Patterns exist but often, to see these patterns, we have to abstract away the mere price flunctuations to find deeper insight.

When a human tries to analyse prices of an asset in a chart, they don't look at the raw prices most times, but rather, an image representing the trend at some time scale. There are visual characteristics of price action.

Once we can abstract these short-term trends and represent them as a snapshot of the market, we can do more interesting work like applying genetic algorithms and decision trees to discovering profitable rule-based strategies.

## Objectives & Milestones
- Train an autoencoder on the images generated from the time-series data 
    - Ensure embeddings are clusterable (Use of techniques like contrastive learning)
    - Cluster embeddings using clustering algorithm like KMeans

- Relabel time-series to create a sequence of tokens relating to the clusters gotten from previous step
    - Explore rule-based algorithms on the data.
        - Run an algorithm that tries to discover patterns that give rise to profitable signals.
    - Treat problem as masked-token prediction problem as we want the model to understand the relationships across assets in the market. If successful, model can still be used as a next-token predictor.
        - If successful and we have a useful short-term future predictor, try to see how it can be used to develop a profitable trading algorithm.


## Next steps

Explore other algorithms we could add and eventually develop a "Mixture of experts" of sorts
Backtest, record metrics and start development of trading system.
Once trading system is developed, deploy live and sit back.