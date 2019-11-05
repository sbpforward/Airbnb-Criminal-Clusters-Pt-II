## Airbnb in Denver: Finding the Criminal Clusters
### Sarah Forward

Building upon my [first capstone](https://github.com/sbpforward/Airbnb-in-Denver) where I conducted an initial exploratory data analysis, I decided to continue the search to determine whether or not I could predict a particular Airbnb listing was in violation of [Denver's Short-Term Rental license](https://www.denvergov.org/content/denvergov/en/denver-business-licensing-center/business-licenses/short-term-rentals/short-term-rental-faq.html).

Quick review: To obtain a Short-Term Rental license in Denver...

+ You must be a legal resident of the United States.
+ **The property must be your primary residence.**
+ **If the property is owned by an LLC, the applicant must be the owner of that of that entity.**
+ If the property is leased, a copy of the lease with the landlord's approval to use the property as a short-term rental
+ Must have some kind of Colorado State approved ID w/ the address along with vehicle regristration, tax documents, or utility bill — all with the same, Colorado address. 
+ You must possess all applicable Business Tax licenses, including the Lodger’s Tax.
+ Tax License Application
+ Check with your insurance carrier to ensure that your STR is covered.

The Data
------
I found my data from [Inside Airbnb](http://insideairbnb.com/about.html). It's owner and moderator, Murray Cox, is an Austrailian-native-turned-New-York-transplant who runs the independent, non-commercial site as a tool to help discover "how Airbnb is really being used in cities around the world."

He's been scraping publicly available data on Airbnb from more than 100 cities around the world. You can get listings, calendar dates and availability, and reviews for cities like Amsterdam, Barcelona, Cape Town, Hong Kong, and, of course, Denver.

There are 4,511 listings and 106 rows in the original dataset. I kept columns that I believed to be most relevant about the listing and the host.

From Unsupervised to Semi-Supervised
------
The dataset does not come with a column that states "Yes" or "No" if the listing is in violation, so I found 100 listings that I understood to be violation Denver's terms. Since I don't have access to legal paperwork, I searched "Host Location" for cities that were **NOT** in Denver. Out of the 200 listings and host profiles I dug through, nearly half were in violation.

Here is a listing:

![alt text](images/example-listing.png "Airbnb Listing")

Here is the host's profile:

![alt text](images/example-profile.png "Profile")

------

Based on my search, I identified 4 kinds of people who are in violation based on their "Host Location". They are...
1. A vacation home rental managment persons/companies.

![alt text](images/twofer-julia.png "General location and a rental management company")

2. Within some kind of driving distance to Denver:
   + Cities like Lakewood or Aurora as "Greater Denver Area", but actually in neighboring counties such as Jefferson, Adams, Arapahoe, or Boulder.
   + Neighboring front range major cities such as Fort Collins and Colorado Springs.
   + Mountain dwellers that live in Aspen, Vail, or Breckenridge. 

![alt text](images/centennial.png "Eventual Denverites")

3. People that don't even live in the same state or country!

   + Discovered 166 total listed cities, 21 of which are not in the United States. There are Denver hosts located in Gibraltar, Spain, United Kingdom, Vietnam, Isreal, Austrailia!

![alt text](images/out-of-country.png "Overseas Host")

4. People who leave their location very broad by entering "Colorado" or "United States" and sometimes nothing at all.

Cleaning the Data
------
+ Listing ID & URL
+ Host ID & Location
+ Host About 
+ Total listings the host has and by type of place
    + Entire home/apt
    + Private room
    + Shared room
+ Details like room type, number of rooms throughout the house
+ Pricing
   + Must have a daily rate, but not all homes set special weekly or monthly rates. 
+ Hot Encoded the following — 1: Yes // 0: No
   + Host location
   + Host Superhost
   + Listing location in Denver?
        + Englewood
        + Lakewood
        + Wheat Ridge
        + Aurora
        + Littleton
        + Stapleton
    + Requires license?
        + Campers/RVs do not
    + If required, is their license information listed AND current?
        + 550 listings out of 4511 are current with "2019" in its designated position


![alt text](images/license-check.png  "Checking the license number")

PCA
-----
Wanting to get a better sense of my data structure and to see what features might be collinear, I used pricncipal component analysis. At first, I used all of the numerical columns in my dataset which... didn't tell me much.

![alt text](images/PCA_All-Features.png  "All Features")

I removed half of the columns which included information I knew to be correlated such as the pricing (daily, weekly, monthly) and the number of bedrooms and bathrooms.

![alt text](images/PCA_Fewer-Categorical-Features.png  "PCA_Fewer Categorical Features")

Although I ended up using all 5 components in my KMeans model, I did check to see using only the first 3 would capture 90% of the explained variance. 

![alt text](images/explained-variance.png  "explained-variance")

KMeans Model
-----
After diving into my PCA, I entered the principal component values in to my KMeans model to see if I could find patterns based on their similiarity. 

![alt text](images/kmeans-clusters.png  "test")

Evaluating the Model
-----
With KMeans, you'd usually check the model using an elbow plot or silhouette score, but I used the 100 target variables I created to help visualize which cluster had the greatest count of Denver's violators.

After playing with the number of clusters, I found 8 to be the sweet spot. When first trying 3 clusters, there was a clear winner but each column was super dense. At 10, it was too spread a litte too thin.

![alt text](images/clusters-by-target.png  "clusters-by-target")

Cluster 5 had the largest weight of violators so my next step was to dig in to see the what these listings were! Just going through a few, I noticed there were a lot more people an empty "Host Location" than those in another city entirely. 

![alt text](images/post-kmeans-hunt.png  "post-kmeans-hunt")

![alt text](images/post-kmeans-hunt-profile.png  "post-kmeans-hunt-profile")


Final Thoughts + Next Steps
-----
In order to get a true accuracy score, I'd have to go through every listing and verify it myself. While I'd love to be able to predict a definitive "yes" or "no" on whether a listing is violating Denver's Airbnb Short-Term Rental regulations, I beleive this could be a good start to pointing Denver's Department of Excise and Licenses on where to start their search.
