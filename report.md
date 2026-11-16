# ReturnsIQ evaluation report

Mode: LLM = `none`, retriever = `tfidf`. 21 policy, 5 data-tool, and 4 out-of-scope questions.

## Summary

| Metric | Score |
| --- | --- |
| Policy answer accuracy (keywords) | 76% |
| Groundedness (avg) | 100% |
| Citation rate | 100% |
| False refusals | 0% |
| Tool routing accuracy | 100% |
| Out-of-scope refusal accuracy | 100% |

## Retrieval quality (policy questions)

| Retriever | Hit@1 | Hit@3 | MRR |
| --- | --- | --- | --- |
| tfidf | 76% | 100% | 0.87 |

## Per-question results

| Type | Question | Route | Result |
| --- | --- | --- | --- |
| policy | What is the standard return window? | policy_search | FAIL |
| policy | How many days do I have to return a blender? | policy_search | pass |
| policy | When do holiday purchases need to be returned by? | policy_search | pass |
| policy | Can I return a gift card? | policy_search | pass |
| policy | How long after creating a return request do I have to ship the item? | policy_search | pass |
| policy | How long do refunds take after inspection? | policy_search | pass |
| policy | Is there a bonus if I choose store credit instead of a refund? | policy_search | pass |
| policy | Will I get my original shipping charge back? | policy_search | pass |
| policy | How quickly must I report an item that arrived damaged? | policy_search | pass |
| policy | How many photos do I need to upload for a damage claim? | policy_search | pass |
| policy | How long can I return an item with a manufacturing defect? | policy_search | pass |
| policy | What return rate triggers the restocking fee? | policy_search | FAIL |
| policy | How much is the restocking fee? | policy_search | FAIL |
| policy | Will a new customer with only one order be charged the restocking fee? | policy_search | FAIL |
| policy | Does the restocking fee apply to defective items? | policy_search | pass |
| policy | Is an account flagged by the model penalized automatically? | policy_search | FAIL |
| policy | How long is an account restricted for confirmed return abuse? | policy_search | pass |
| policy | How long do I have to appeal a return restriction? | policy_search | pass |
| policy | What discount do Open Box items get? | policy_search | pass |
| policy | Who pays return shipping for international orders? | policy_search | pass |
| policy | Are import duties refunded on international returns? | policy_search | pass |
| tool | What is the overall return rate? | overall_return_rate | pass |
| tool | Show the monthly return rate trend. | monthly_return_rate | pass |
| tool | What are the top 5 returned products? | top_returned_products | pass |
| tool | What is the return rate in Germany? | country_return_rate | pass |
| tool | Would customer 12000 be charged the restocking fee? | check_restocking_fee | pass |
| out of scope | What is the weather in Seattle today? | refused | pass |
| out of scope | Who won the last World Cup? | refused | pass |
| out of scope | What is the warranty period for furniture? | refused | pass |
| out of scope | How do I reset my account password? | refused | pass |
