# writing-eval
What if we could test LLMs based on how similar their generated text is to your favorite thought leaders? Or a random college essay? This is meant to be an open-source benchmarking test meant to evaluate the quality of LLM responses.

Right now, this is a very rough POC. The data is stored in a csv file with rows indexing the length, text, and tokens. I essentially used a collection of Paul Graham's 219 essays, along with some of my favorite Sam Altman essays, along with some essays found  <a href = "https://ivypanda.com/essays/">here</a>  . You can adjust the code to use an alternative framework such as SQL-Lite or some other database. You are also free to edit the code to use any data that you may like. I tested this primarily with the new Falcon GPT-4 model on Hugging Face, though if you want to use an API, you can probably edit the code for that purpose. 

Current tests include: 
Levenshtein Similarity
Jaccard Similarity
Cosine Similarity
KL Divergence
Euclidean Distance

Feel free to add more. I am probably going try to put all of these tests into a random forest for better visualization, though, right now, most of the tests don't yield anything meaningful. More research is needed on better metrics to compare the quality of the writing with the input. 
