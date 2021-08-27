from emolex_aux import categorize,synthesize

# temp = categorize("28_out_emolex_8_pt.csv")
result = synthesize("28_categorized_tweets_8_eng.csv","28")
print(result)
result.to_csv("28eng_synthesized.csv")