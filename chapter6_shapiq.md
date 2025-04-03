## SHAPIQ (2024)

SHAPIQ is a Python library designed for computing and analyzing Shapley Interactions (SIs) of any order, which can reveal synergies and redundancies between entities in machine learning tasks. It provides a general interface for state-of-the-art SI algorithms, an explanation API for interpreting model predictions using SIs, and tools for benchmarking and visualizing these interactions across various machine learning domains, offering a more comprehensive understanding of feature relationships than traditional Shapley Values.

### Cooperative Game with Three Players

Consider a cooperative game with three players. So, set $N = \{1, 2, 3\}$.
And let the value function $ v(S) $ define the payout for each possible coalition $S \subseteq N$.
So the possible combinations for the game are as follows:

- **Order 0 interactions:** $ \{\} $
- **Order 1 interactions:** $ \{1\}, \{2\}, \{3\} $
- **Order 2 interactions:** $ \{1, 2\}, \{2, 3\}, \{1, 3\} $
- **Order 3 interactions:** $ \{1, 2, 3\} $

With SHAPIQ, we are interested in the calculation of interactions like these: $ \{1, 2\}, \{2, 3\}, \{1, 3\}, \{1, 2, 3\} $.

### Interaction Index Representation
Interaction Index is represented by:

$$ \Phi^{II}(S) = \sum_{T\subseteq N \setminus S }^{} p^{|S|}_{|T|}(n) \Delta_{S}(T) $$

Here,

$$ p^{|S|}_{|T|}(n) = \frac{1}{(n-s+1) (\binom{n-s}{t})} $$

$$ \Delta_{S}(T) = \textnormal{Discrete derivative for } S \subseteq N \textnormal{ in the presence of } T \subseteq N \setminus S $$

### Interaction Index Calculation for $ \{1, 2\} $

$$ \Phi^{II}(\{1, 2\}) = \sum_{T\subseteq \{1, 2, 3\} \setminus \{1, 2\} }^{} \frac{1}{(3-2+1)(\binom{3-2}{t})} \Delta_{\{1, 2\}}(T) $$

Here, the possible cases of $T$ are $\{\}$ and $\{3\}$.

So,

$$ \Phi^{II}(\{1, 2\}) = \frac{1}{(3-2+1)(\binom{3-2}{0})} \Delta_{\{1, 2\}}(\{\}) + \frac{1}{(3-2+1)(\binom{3-2}{1})} \Delta_{\{1, 2\}}(\{3\}) $$

#### Discrete Derivative Calculation

$$ \Delta_{\{1, 2\}}(\{\}) = \sum_{L\subseteq S }^{} (-1)^{|S|-|L|} v(\{T \cup L\}) $$

$$ \Delta_{\{1, 2\}}(\{\}) = \sum_{L\subseteq S }^{} (-1)^{2-|L|} v(\{\{\} \cup L\}) $$

Possible values for $L$ are $\{\}, \{1\}, \{2\}, \{1, 2\}$.

$$ \Delta_{\{1, 2\}}(\{\}) = v(\{\}) -  v(\{1\}) - v(\{2\}) + v(\{1, 2\}) $$

For $ \Delta_{\{1, 2\}}(\{3\}) $:

$$ \Delta_{\{1, 2\}}(\{3\}) = \sum_{L\subseteq S }^{} (-1)^{|S|-|L|} v(\{T \cup L\}) $$

$$ \Delta_{\{1, 2\}}(\{3\}) = v(\{3\}) -  v(\{1, 3\}) - v(\{2, 3\}) + v(\{1, 2, 3\}) $$

### Final Interaction Index Formula

$$ \Phi^{II}(\{1, 2\}) = \frac{1}{2} \left( v(\{\}) -  v(\{1\}) - v(\{2\}) + v(\{1, 2\}) \right) + \frac{1}{2} \left( v(\{3\}) -  v(\{1, 3\}) - v(\{2, 3\}) + v(\{1, 2, 3\}) \right) $$

### Value Function Definition

Now, we need to get the values from the value function for the terms:
$ v(\{\}), v(\{1\}), v(\{2\}), v(\{3\}), v(\{1, 2\}), v(\{2, 3\}), v(\{1, 3\}), v(\{1, 2, 3\}) $.

Here, $v$ is the value function representing the worth of a coalition (e.g., the model's prediction with a subset of features).

- $v(\{\})$: The value for the empty coalition, meaning it holds the value when no features are added. This is the model’s base value or the prediction without any feature information.
- For any other coalition $v(\{S\})$, it is obtained by evaluating the model (or its conditional expectation) on the input where only the features in $S$ are present, and other features are handled (e.g., by marginalization or conditioning).