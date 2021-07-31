---
layout: article
title: "[Linear Regression (1)] 표본 평균과 표본 분산"
subtitle: 표본 평균과 표본 분산
categories: statistics 
date: 2021-07-31T12:36:00+09:00
tags: statistics regression
comments: true
---
## 개요
 Linear Regression에서 가설 검정시 사용되는 표본 평균과 표본 분산에 대해서 알아본다.

## 표본 평군과 표본 분산

임의의 Random Variable $X$가 평균을 $\mu$, 표준편차를 $\sigma$를 따를 때 (no specific distribution),

$$
X\sim{}\mbox{unknown}(\mu, \sigma^2)
$$

### 표본 평균($\bar{X}$)

**표본평균의 정의**

임의의 Random Variable의  임의의 n개 표본에 대한 평균을 일컬음.


$$
\frac{1}{n}\overset{n}{\underset{i=1}{\sum}}X_i
$$


**표본 평균의 기대값**


$$
\mathbb{E}(\bar{X})=\mathbb{E}(\frac{1}{n}\overset{n}{\underset{i=1}{\sum}}X_i)=\frac{1}{n}\mathbb{E}(\overset{n}{\underset{i=1}{\sum}}X_i) = \frac{1}{n}*n*\mu = \mu
$$

**표본 평균의 분산**


$$
Var(\bar{X}) = Var(\frac{1}{n}\overset{n}{\underset{i=1}{\sum}}X_i) = \frac{1}{n^2}Var(\overset{n}{\underset{i=1}{\sum}}X_i) = \frac{\sigma^2}{n}
$$

### 표본 분산($s^2$)

**표본 분산의 정의**

임의의 Random Variable의  임의의 n개 표본에 대한 평균을 일컬음.


$$
s^2 = \frac{1}{n-1}\overset{n}{\underset{i=1}{\sum}}(X_i - \bar{X})^2
$$


**왜 표본분산은 표본 평균과 다르게 n-1로 나눌까?**



**Proof**



만약 표본 분산을 $\bar{s}^2=\frac{1}{n}\sum(X_i-\bar{X})^2$와 같이 표본의 개수 $n$만큼 분모로 나눈다고 가정해보자.

그렇다면,


$$
\begin{aligned}
\mathbb{E}(\bar{s}^2)&=\mathbb{E}(\frac{1}{n}\sum(X_i-\bar{X})^2)\\
&=\mathbb{E}(\frac{1}{n}\sum(X_i^2-X_i\bar{X}-\bar{X}X_i+\bar{X}^2))\\
&=\mathbb{E}(\frac{1}{n}\sum{X_i^2}-\frac{1}{n}2\bar{X}\sum{X_i}+\frac{1}{n}\sum{\bar{X}^2})\\
&=\mathbb{E}(\frac{1}{n}\sum{X_i^2}-2\bar{X}^2+\frac{1}{n}n\bar{X}^2)\\
&=\mathbb{E}(\frac{1}{n}\sum{X_i^2})-2\mathbb{E}(\bar{X}^2)+\mathbb{E}(\bar{X}^2)\\
&=\frac{1}{n}\sum{\mathbb{E}(X_i^2)}-\mathbb{E}(\bar{X}^2)\\
&=\frac{1}{n}(\mathbb{E}(X_1)^2+\mathbb{E}(X_2)^2+...+\mathbb{E}(X_n)^2)-\mathbb{E}(\bar{X}^2)\\
&=\frac{1}{n}((\sigma^2+\mu^2)+(\sigma^2+\mu^2)+...+(\sigma^2+\mu^2))-(\frac{\sigma^2}{n}+\mu)\ (\because{}\mbox{분산의 성질})\\
&=\frac{1}{n}*n*(\sigma^2+\mu^2)-\frac{\sigma^2}{n}-\mu^2\\
&=\sigma^2 - \frac{\sigma^2}{n} \\
&=\frac{n-1}{n}\sigma^2
\end{aligned}
$$



#### 분산의 성질

$$
\begin{aligned}
Var(X) &= \mathbb{E}[(X-\mu)^2] \\
&= \mathbb{E}(X^2)-\mu^2
\end{aligned}
$$



따라서, 

$$Var(X)=\mathbb{E}(X^2) - (\mathbb{E}(X))^2 \\ \mathbb{E}(X^2)=Var(X)-(\mathbb{E}(X))^2 = \sigma^2 + \mu^2$$

$$Var(\bar{X})=\mathbb{E}(\bar{X}^2) - (\mathbb{E}(\bar{X}))^2 \\ \mathbb{E}(\bar{X}^2)=Var(\bar{X})-(\mathbb{E}(X))^2 = \frac{\sigma^2}{n} + \mu^2$$



$\bar{s}^2$의 기대값은 모수의 분산보다 작음을 알 수 있다.  

그런데, 우리는 표본의 분포로 모수를 추정하고자 한다. 따라서, $\bar{s}^2$ 대신에, $\frac{n}{n-1}\bar{s}^2$을 표본 분산으로 사용한다면, 이것의 기댓값은 모수와 같은 $\sigma^2$가 될 것이다.
$$
\begin{aligned}
\mathbb{E}(\frac{n}{n-1}\bar{s}^2) &= \frac{1}{n}*\frac{n}{n-1}\sum(X_i-\bar{X})^2\\
&=\frac{1}{n-1}\sum{(X-\bar{x})^2}
\end{aligned}
$$



따라서, $\frac{n}{n-1}\bar{s}^2$ replace with $s^2$ as sample variance.


$$
\therefore{} s^2=\frac{1}{n-1}\sum{(X_i-\bar{X})^2}
$$


## 출처

1. <a herf="https://teamdable.github.io/techblog/Sample-Mean-and-Sample-Variance">Sample Mean and Sample Variance</a>
2. Rencher, Alvin C., and G. Bruce Schaalje. *Linear models in statistics*. John Wiley & Sons, 2008.