### KING_COUNTY_PROJECT ###

*For this project we asked the following questions:*
1. What is the impact of renovation before selling a house?
2. What is the best time to sell a house?
3. What are the best indicators of price?
4. Using those indicators can we make further recommendations?

Data: we use provided data for King County that can be found on Kaggle (that has been modified by flatiron). 

To answer the first question we graph the difference between prices of home which have renovation vs. the houses which are not.
We also calculate the percent difference renovation can make.  

To answer our second question we determine the best month and season that houses are most often sold. 

To answer our third question we use a multi-linear regression model to get best indicators with at least 80 percent accuracy. 
To extend our third question: we find the best indicators are location and size of house so we go ahead and find what are the best locations by zip code. 
We also make a recommendation that it is best to buy a house that is larger but not an outlier, for example a 33 bedroom house is larger but we do not recommend buying such a house. Instead we recommend, given a 2-3 bedroom house one that has more square footage than the average. 

All visualizations are made by matplotlib and seaborn
# module2_project
