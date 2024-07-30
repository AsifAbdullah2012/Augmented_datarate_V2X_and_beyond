
# Augmented data rate in V2X Using ML

###### Some Initial code base has been shared for experimental purpose. Main code base follows a cleaner version with a framework structure will not be published for organizational restrictions. 




## Background
This paper addresses the challenge of Modulation and Coding Scheme (MCS) selection in C-V2X sidelink communication, especially in the absence of a feedback channel. With increasing demands for higher data rates in vehicle-to-everything (V2X) applications, the paper proposes a machine learning (ML) approach to predict optimal MCS levels for maximizing data rates. The methodology employs quantile prediction combined with various ML algorithms, demonstrating significant improvements over traditional methods. A unique dataset from real-world drive tests is provided to facilitate further research.

## Problem Statement
In V2X communication, particularly under Mode 4 (out-of-coverage operation), User Equipments (UEs) autonomously select transmission parameters, including MCS levels. The absence of a feedback mechanism complicates optimal MCS selection, crucial for applications requiring high data rates, such as video streaming and enhanced situational awareness. Traditional methods often result in suboptimal performance, prompting the need for adaptive and predictive approaches.

## Methodology
The authors introduce a machine learning framework for MCS prediction, focusing on achieving the highest possible data rate. The proposed solution includes:

1. **Data Collection**: A comprehensive dataset was gathered through extensive drive tests, encompassing various environments and conditions.
2. **Feature Extraction**: Key features influencing MCS selection were identified, such as signal strength, interference levels, and geographical context.
3. **ML Models**: Several machine learning algorithms, including regression and classification models, were trained using the extracted features to predict the optimal MCS level.
4. **Quantile Prediction**: To enhance robustness, the approach incorporates quantile prediction, estimating the probability distribution of achievable data rates for each MCS level.

## Results
The proposed machine learning-based method significantly outperformed conventional approaches in predicting suitable MCS levels, leading to higher data rates. The evaluation metrics demonstrated substantial gains in throughput, with reduced prediction errors and improved adaptability to varying channel conditions.

## Significance
The study provides a viable solution to the critical issue of MCS selection in V2X sidelink communication, particularly for scenarios lacking a feedback channel. By leveraging machine learning, the approach facilitates more efficient use of available spectrum, supports higher data rate applications, and enhances the overall performance of V2X systems. The publicly available dataset also serves as a valuable resource for the research community, promoting further advancements in the field.

## Conclusion
The integration of machine learning techniques in sidelink communication presents a promising direction for optimizing MCS selection. The results affirm the potential for significant improvements in data rates and system performance, essential for meeting the demands of emerging V2X applications.

## Acknowledgement
This research was partially funded by the Federal Ministry of Education and Research (BMBF) of Germany under the AI4Mobile (16KIS1170K) and 6G-RIC (16KISK020K) projects.
