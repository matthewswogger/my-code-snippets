# my-code-snippets
A dumping ground for useful code chunks to keep around

## ngrok
To work with ngrok for development check out the `README.md` in the `/ngrok` directory.

## Time to Pay Invoice prediction and Cash Flow Forecasting
This is an interesting method I developed for predicting when an invoice (or anything really) will be paid.

It is based on the [Exponential Distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
and the mathmatical reality that the predicted propabilities of any Multivariate Classification Algorithm
are in fact a [PMF](https://en.wikipedia.org/wiki/Probability_mass_function).
Which can be easily turned into a [CDR](https://en.wikipedia.org/wiki/Cumulative_distribution_function).
Which means you can create a 'sudo' Exponential Distribution using any classification algorithim by designing the target variable correctly.

To work with this check out `/invoice_prediction` directory.
