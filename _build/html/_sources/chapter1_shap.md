## SHAP (2017)

The SHAP framework identifies a unique solution within the class of additive feature attribution methods, based on game theory results, that satisfies a set of desirable properties: local accuracy, missingness, and consistency. SHAP values are proposed as a unified measure of feature importance, defined as the Shapley values of a conditional expectation function of the original model. They represent the change in the expected model prediction when conditioning on a particular feature, explaining how to get from the base value (expected prediction without knowing any features) to the current output.

### Cooperative Game with Three Players

Consider a cooperative game with three players. So, set $N = \{1, 2, 3\}$.
And let the value function $ v(S) $ define the payout for each player $ \{1\}, \{2\}, \{3\} $.

### Shapley Value Calculation
Shapley Value is represented by:

$$ \Phi_{i} = \sum_{S\subseteq F \setminus \{i\} }^{}  \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_{S\cup\{i\}}(x_{S\cup\{i\}} - f_{x}(x_{S}))]$$

Let's calculate $ \Phi_{\{1\}} $ for the game where the players are represented by $ F = \{1,2,3\} $ and the payout function is $ v $.

So,

$$ \Phi_{\{1\}} = \sum_{S\subseteq \{1,2,3\} \setminus \{1\} }^{}  \frac{|S|!(3-|S|-1)!}{3!} [v(S\cup\{1\}) - v(S)]$$

Here, S can be $ \{2,3\}, \{2\}, \{3\}, \{\} $

So, 

$$ \Phi_{\{1\}} = \frac{|2|!(3-2-1)!}{3!} [v(\{1, 2, 3\}) - v(\{2, 3\})] + \frac{1!(3-1-1)!}{3!} [v(\{1, 2\}) - v(\{2\})] + \frac{1!(3-1-1)!}{3!} [v(\{1, 3\}) - v(\{3\})] + \frac{0!(3-0-1)!}{3!} [v(\{1\}) - v(\{\})] $$

$$ \Phi_{\{1\}} = \frac{1}{3} [v(\{1, 2, 3\}) - v(\{2, 3\})] + \frac{1}{6} [v(\{1, 2\}) - v(\{2\})] + \frac{1}{6} [v(\{1, 3\}) - v(\{3\})] + \frac{1}{3} [v(\{1\}) - v(\{\})] $$

$$ \Phi_{\{1\}} = \frac{1}{3} v(\{1, 2, 3\}) - \frac{1}{3} v(\{2, 3\}) + \frac{1}{6} v(\{1, 2\}) - \frac{1}{6} v(\{2\}) + \frac{1}{6} v(\{1, 3\}) - \frac{1}{6} v(\{3\}) + \frac{1}{3} v(\{1\}) - \frac{1}{3} v(\{\}) $$

$$ \Phi_{\{1\}} = - \frac{1}{3} v(\{\}) + \frac{1}{3} v(\{1\}) - \frac{1}{6} v(\{2\}) - \frac{1}{6} v(\{3\}) + \frac{1}{6} v(\{1, 2\}) - \frac{1}{3} v(\{2, 3\}) + \frac{1}{6} v(\{1, 3\}) + \frac{1}{3} v(\{1, 2, 3\})  $$

### Value Function Definition

Now, we need to get the values from the value function for the terms:
$ v(\{\}), v(\{1\}), v(\{2\}), v(\{3\}), v(\{1, 2\}), v(\{2, 3\}), v(\{1, 3\}), v(\{1, 2, 3\}) $.

Here, $v$ is the value function representing the worth of a coalition (e.g., the model's prediction with a subset of features).

- $v(\{\})$: The value for the empty coalition, meaning it holds the value when no features are added. This is the model’s base value or the prediction without any feature information.
- For any other coalition $v(\{S\})$, it is obtained by evaluating the model (or its conditional expectation) on the input where only the features in $S$ are present, and other features are handled (e.g., by marginalization or conditioning).