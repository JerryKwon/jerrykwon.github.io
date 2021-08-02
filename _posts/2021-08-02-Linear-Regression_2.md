---
layout: article
title: "[Linear Regression (2)] 카이제곱(chi-squared) 분포와 표본 분산"
subtitle: 카이제곱분포와 표본 분산
categories: statistics 
date: 2021-08-02T17:54:00+09:00
tags: statistics regression chi-squared
comments: true
---
## 개요
 Linear Regression에서 검정시 모분산을 모르는 경우 사용되는 t-분포를 정의하는데 사용되는 카이제곱 분포에 대해서 살펴보고, 이를 앞서 배운 표본 분산이 어떤 분포를 가지는지 살펴보자

## 카이제곱(chi-squared) 분포

임의의 정규분포를 따르는 Random Variable $Z$가 평균을 $0$, 표준편차를 $1$를 따르며(표준정규분포), i.i.d(independent identical distributed)를 따르는 경우.

$$
Z\sim{}N(\mu, \sigma^2)
$$

n개의 random variable에 대한 제곱의 합이 따르는 분포를 $\Chi^2(n)$, 카이제곱(chi-squared) 분포라고 정의한다.
$$
Z_1^2+Z_2^2+...+Z_n^2\sim\Chi^2(n) = \overset{n}{\underset{i=1}{\sum}}Z_i^2
$$
이때, n을 자유도(degree of freedom)이라고 하며, 이는 임의의 Sample에서 변수를 자유롭게 정의할 수 있는 정도를 나타낸다.

i.e)

표본평균: $\bar{X}=\frac{1}{n}\underset{i=1}{\overset{n}{\sum}}X_i$; 자유도: $n$

표본분산: $s^2=\frac{1}{n-1}\underset{i=1}{\overset{n}{\sum}}(X_i-\bar{X}^2)$; 자유도: $n-1$

왜냐하면, 잔차의 합은 0이 되어야 하는 제약이 존재하기 때문에, 이를 만족하기 위해 전체 n개의 샘플 중 하나의 샘플은 이를 위해 사용되어야 한다. 그러므로, 자유롭게 설정할 수 있는 변수의 개수는 $n-1$이기 때문에 자유도가 위와 같이 나타난다.

**잔차(residual)**: 변수 - 통계량; **오차(error)**: 변수 - 모수



카이제곱 분포는 표준정규분포를 따르는 random variable에 대해서만 정의되고, 이를 정규화한 형태로 나타내면,
$$
\frac{\sum(Z_i-0)^2}{1}\sim\Chi^2(n)
$$
위와 같아지고, 만약 표준정규분포를 바로 따르지 않는 임의의 정규분포를 따르는 확률변수 $X\sim{}N(\mu,\sigma^2)$가 정의되어 있는 경우.
$$
\begin{aligned}
Z&=\frac{X-\mu}{\sigma}\\
Z^2&=\frac{(x-\mu)^2}{\sigma^2}\\
\underset{i=1}{\overset{n}{\sum}}Z_i^2&=\frac{(X-\mu)^2}{\sigma^2}
\end{aligned}
$$

정규화하여 제곱하는 경우 위와 같이 나타낼 수 있으며, 정규화된 확률변수의 제곱의 합은 카이제곱분포를 따른다고 말할 수 있다.

## 모수에서 통계량으로

그러나, 분포의 모평균은 일반적으로 알고 있지 않기 때문에, 이를 표본 평균으로 대체하면, 어떤 분포를 따르게 될까?
$$
\frac{\sum(X-\bar{X})^2}{\sigma^2}\sim{}\Chi^2(?)
$$
결과는 $n-1$의 자유도를 가지는 카이제곱분포가 된다는 것이다. $\Chi^2(n-1)$



### Proof

$$
\begin{aligned}
&\sum(\frac{X_i-\bar{X}}{\sigma})^2 \\
&=\sum(\frac{X_i-\mu+\mu-\bar{X}}{\sigma})^2\\
&=\sum(\frac{X_i-\mu}{\sigma}+\frac{\mu-\bar{X}}{\sigma})^2\\
&=\sum((\frac{X_i-\mu}{\sigma})^2+(\frac{\mu-\bar{X}}{\sigma})^2+2(\frac{X_i-\mu}{\sigma})(\frac{\mu-\bar{X}}{\sigma}))\\
&=\sum((\frac{X_i-\mu}{\sigma})^2+(\frac{\mu-\bar{X}}{\sigma})^2)+2\sum(\frac{X_i-\mu}{\sigma})\sum(\frac{\mu-\bar{X}}{\sigma})\\
&=\sum(\frac{X_i-\mu}{\sigma})^2+\sum(\frac{\mu-\bar{X}}{\sigma})^2\ (\because{\mbox{sum of residual is 0}}) \\
&\therefore{}\sum(\frac{X_i-\mu}{\sigma})^2=\sum(\frac{\mu-\bar{X}}{\sigma})^2+\sum(\frac{X_i-\bar{X}}{\sigma})^2
\end{aligned}
$$

여기서, 표본 분산의 식을 활용하게 되면,
$$
s^2=\frac{1}{n-1}\sum(X_i-\bar{X})^2\\
\therefore{}\overset{n}{\underset{i=1}{\sum}}(X_i-\bar{X})^2=(n-1)s^2
$$
위의 전개한 결과를 Proof의 결과에 대입하게 되면,
$$
\sum(\frac{X_i-\mu}{\sigma})^2=\frac{(n-1)s^2}{\sigma^2}+\sum(\frac{X_i-\bar{X}}{\sigma})^2\\
\mbox{Again, }\frac{\sum(X_i-\mu)^2}{\sigma^2} = \frac{(n-1)s^2}{\sigma^2}+\frac{n(\bar{X}-\mu)^2}{\sigma^2}
$$

#### Proof 1 - 카이제곱분포의 성질을 이용

임의의 두 카이제곱 분포가 있고, 두 확률변수가 독립이라면, 아래와 같은 성질을 가진다.

$Z\sim{}N(0,1)$,  $Z^{\prime}\sim{}N(0,1)$ and $Z\mbox{ and }Z^{\prime}\mbox{ independent}$.

then, $\Chi^2(n)+\Chi^2(m)=\Chi^2(n+m)$

위를 이용하여 살펴보면, 좌변은 n의 자유도를 가지는 카이제곱분포이다. 그리고 우변의 두번째 변수는 1의 자유도를 가진다.

그렇다면, 우변의 첫번째와 두번째 변수에 사용되는 표본 분산과 표본 평균이 독립이라면, 성질에 의해 우변의 첫번째 변수는 $\Chi^2(n-1)$ n-1의 자유도를 가지는 카이제곱 분포를 따를 것이다.



**Independence btw $\bar{X}$ and $s^2$**

When, $X\sim{}N_p(\mu,\sigma^2I)$
$$
\bar{X}=\frac{1}{n}j^TX\\
\bar{X}^2=(\frac{1}{n})^2X^TJX\\
n\bar{X}^2=X^T\frac{1}{n}JX=\bold{X^TAX}
$$

$$
\begin{aligned}
s^2&=\frac{1}{n-1}\sum(X_i-\bar{X})^2\\
&=\sum{X_i}^2-n\bar{X}^2\\
&=X^TIX-X^T\frac{1}{n}JX\\
&=X^T(I-\frac{1}{n}J)X\\
&=\bold{X^TBX}
\end{aligned}
$$

By Linear Models in Statistics. Theorem 5.6b. Collary 1,

$y^TAy$ and $y^TBy$ are independent if and only if AB=0.

our A and B is $A=\frac{1}{n}J$, $B=I-\frac{1}{n}J$

So, $AB=\frac{1}{n}J(I-\frac{1}{n})J=-\frac{(n-1)}{n^2}+\frac{n-1}{n^2}=0$



따라서, 표본 평균과 표본 분산은 독립이라고 말할 수 있으며, 이로 인해 표본 분산의 분포는 카이제곱분포를 따른다고 할 수 있다.