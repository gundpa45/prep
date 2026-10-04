# Student Performance Prediction System: Viva Preparation

**1. What is Machine Learning?**  
*Answer:* Machine learning is a subset of artificial intelligence that involves training a computer system using historical data to recognize patterns and make decisions or predictions on new data without being explicitly programmed for every scenario.

**2. Why is this a regression problem?**  
*Answer:* The target variable in this project is `G3` (final examination score), which is a continuous numerical value ranging from 0 to 20 rather than a category (like "Pass" or "Fail"). Hence, a regression approach is required to predict the exact numerical score.

**3. Why did you choose Random Forest as your final model?**  
*Answer:* Random Forest operates as an ensemble of multiple decision trees, which significantly reduces the variance often causing single decision trees to overfit the training data. Given our 33 features—many of them highly non-linear categorical data—Random Forest handles these interactions best and achieved the highest R² metric on unseen test data.

**4. What is Linear Regression?**  
*Answer:* A linear regression model assumes a linear relationship between the input variables and the single continuous output variable. It tries to draw a straight line of best fit that minimizes the residuals (errors) between the predicted and actual data points.

**5. What is a Decision Tree?**  
*Answer:* A Decision Tree creates a flowchart-like structure where each internal node represents a logical test on an attribute, every branch represents an outcome of the test, and the leaf nodes represent the final predicted value (in regression).

**6. What is a Random Forest?**  
*Answer:* Random Forest is an ensemble learning method constructed by building multiple decision trees during training via a process called "bagging" and outputting the mean prediction of the individual trees, preventing overfitting.

**7. What is Gradient Boosting?**  
*Answer:* Gradient Boosting is a machine learning technique that trains weak models (typically shallow decision trees) sequentially. Each consecutive tree attempts to correct the errors (residuals) made by the combination of all previously built trees.

**8. What is Train/Test split and why do we do it?**  
*Answer:* It involves dividing the dataset into two distinct partitions, usually an 80/20 split. The larger train set is used exclusively for training the model. The test set acts as unseen future data to impartially evaluate whether the model has generalized its learning rather than just memorizing the past data.

**9. What is Overfitting?**  
*Answer:* Overfitting occurs when a machine learning model learns the training dataset too perfectly, including all its noise and outliers. Consequently, it performs exceptionally well on training data but poorly on the unseen test data.

**10. What is Underfitting?**  
*Answer:* Underfitting happens when the model is too simple to capture the underlying structure of the data, resulting in poor performance on both the training data and unseen data. 

**11. What is Data Leakage?**  
*Answer:* Data leakage occurs when information from outside the training dataset (such as calculating scaling means across the entire dataset before splitting) is used to create the model, producing overly optimistic test results yielding an invalid model.

**12. What is MAE (Mean Absolute Error)?**  
*Answer:* MAE is the average absolute difference between the predicted values and the actual values. It measures average error magnitude without considering error direction.

**13. What is MSE (Mean Squared Error)?**  
*Answer:* MSE takes the average of the squared differences between predicted and actual values. Squaring the errors penalizes larger errors much heavier than smaller errors.

**14. What is RMSE (Root Mean Squared Error)?**  
*Answer:* RMSE is simply the square root of MSE. It translates the error metric back into the exact same units as the target variable (points out of 20), making it much easier for humans to interpret.

**15. What is the R² (Coefficient of Determination) Score?**  
*Answer:* The R² score represents the proportion of variance in the target variable that is smoothly predictable from the independent features. A score of 1.0 represents perfect predictions whereas 0 indicates predicting the baseline mean. Our model yielded roughly 81% prediction success variance capability (R² = 0.81).

**16. Why is Data Preprocessing required?**  
*Answer:* Real-world datasets often possess missing values, string-based categories, varying scales, and noise. Machine learning models predominantly depend solely on numerical matrices, making preprocessing critical.

**17. Why do we encode categorical variables?**  
*Answer:* Regression algorithms require raw numerical inputs. Since variables like "Mother's Occupation" are text strings, we use One-Hot Encoding to convert them into a matrix of dense 0s and 1s which mathematical algorithms can successfully process.

**18. Why do we use Streamlit?**  
*Answer:* Streamlit is an open-source framework perfectly suited for rapidly turning Python data scripts into shareable web applications via its highly simplified frontend API layout techniques. It removes massive integration overhead required to hook ML code up to custom HTML/React stacks.

**19. What are the limitations of this project?**  
*Answer:* Predictive limitations include that this model relies heavily on internal marks assessment features which occur later in the academic year, rendering early-term warnings slightly less accurate. It also fundamentally treats human learning solely through metric correlation without addressing deeper psychological drivers.

**20. What is the future scope of this research?**  
*Answer:* Future expansions would involve linking this architecture into digital Learning Management Systems containing live student engagement telemetry, and upgrading the prediction module utilizing complex deep neural networks given much larger multi-school datasets.
