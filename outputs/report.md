# BFCL-Style Held-Out Evaluation Report

## Summary

- Held-out examples: 128

- Baseline (post-SFT, pre-GRPO) accuracy: 28.1%

- Post-GRPO accuracy: 33.6%

- Delta: +5.5pp


## Held-out accuracy during GRPO training (checked every chunk, not just at the end)

- step 0 (baseline): 28.1%

- step 25: 31.2%

- step 50: 30.5%

- step 75: 32.0%

- step 100: 32.8%


## Sample predictions (post-GRPO)

- [incorrect] "Hey, Calculate the tip for a 100.84 bill at 90.22%." -> `calculator_tip (bill_total = 244.95, tip_percent = 83.83)` (expected `calculator_tip`)

- [correct] "Schedule an event titled Budget laptops for 2026-05-22." -> `calendar_create_event (title = 'Budget laptops', date = '2026-05-22')` (expected `calendar_create_event`)

- [correct] "Hey, List my calendar events for 2026-05-22." -> `calendar_list_events (date = '2026-05-22')` (expected `calendar_list_events`)

- [correct] "Could you execute the following python snippet: python tutorials." -> `code_execute (language = 'python', code = 'python tutorials')` (expected `code_execute`)

- [correct] "Could you find Raj in my contacts." -> `contact_lookup (name = 'Raj')` (expected `contact_lookup`)

- [incorrect] "Could you how much is 213.18 GBP in USD?" -> `currency_convert (amount = 120.95, from_currency = 'GBP', to_currency = 'USD')` (expected `currency_convert`)

- [incorrect] "Hey, Look up rows in electric vehicle reviews matching python tutorials." -> `database_query (table = 'python tutorials', filter = 'electric)` (expected `database_query`)

- [correct] "Could you email Tom about Best pizza nearby." -> `email_send (to = 'Tom', subject = 'Best pizza nearby')` (expected `email_send`)

- [incorrect] "Quick favor -- Search for files named like weekend hiking trails." -> `hotel_search (title = 'Best pizza nearby')` (expected `file_search`)

- [incorrect] "Quick favor -- Find a flight from London to Mumbai for 2026-04-05." -> `flight_search (origin = 'London', destination = '2026-04-05')` (expected `flight_search`)
