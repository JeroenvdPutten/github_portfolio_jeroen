# Learn by tracing one review

## 1. The model

Read `app/model.py`. Before running it, answer:

- What is downloaded, and what happens locally?
- Why do we load the model once instead of per review?
- How are characters different from tokens?

Try: "Excellent product", "Terrible product", "Not a terrible product", and
"Excellent, another broken charger". Predict the output before submitting each.
Find a wrong prediction. Does the model score warn you about the mistake?

## 2. The API

Open http://localhost:8000/docs and expand `POST /predict`.
Use **Try it out** to submit a review, then submit an empty string.
Compare the HTTP status codes and response bodies.

Read `app/schemas.py` and `app/api.py`.
Explain where validation ends and prediction begins. Why is 422 useful to a client?
Why does the batch response preserve order?

## 3. The dashboard

Read `post_prediction` in `dashboard.py`. Follow the request URL, JSON body,
timeout, response, and error handling. The dashboard has no model instance.

Upload `examples/reviews.csv`. Can you recreate the positive percentage from
the rows in the table? Change a CSV header to `review` and see the validation message.

## 4. Evaluation

Generate the evaluation report and open `notebooks/01_evaluation.ipynb`.
Accuracy is the fraction of correct predictions. Precision asks how often a predicted
class was correct. Recall asks how many actual examples of that class were found.
F1 balances precision and recall.

Find three errors in `reports/errors.csv`. Categorize why they happened: mixed sentiment,
sarcasm, star-rating disagreement, truncation, or something else. Check their scores.
Why is an always-negative baseline 50% here? Would it still be 50% for a real webshop?
What happens if we repeatedly adjust the model based on these same 500 reviews?

## 5. Docker

Read `Dockerfile`, then `compose.yaml`. Explain image versus container in your own words.

Run `docker compose ps`: why are there two running containers but one application image?
Why does the dashboard use `http://api:8000` while your browser uses `localhost`?
What survives `docker compose down`, and why?

## Portfolio explanation

Practice a two-minute explanation covering the business question, dataset,
pretrained model, evaluation versus baseline, API, dashboard, and one limitation.
Be precise about what you built and what you reused.
