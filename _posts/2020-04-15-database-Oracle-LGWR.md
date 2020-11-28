---
layout: post
title: "[Oracle] LGWR 동작 방식"
subtitle: Oracle LGWR 동작 방식
categories: database
date: 2020-04-15T15:45:00+09:00
tags: Oracle LGWR 
comments: true
---

## 참고자료

1. Jonathan Lewis, 번역. (주)엑셈 - Oracle Core Essential Internals for DBAs and Developers

2. DBA 커뮤니티 구루비 - '6장 기록과 복구'

   http://wiki.gurubee.net/pages/viewpage.action?pageId=28607002
   
   

## 리두 로그 버퍼

<img src="http://wiki.gurubee.net/download/attachments/28607002/6-1.png" />

 리두 로그 버퍼는 Oracle에서 SGA내에 있는 변경된 데이터 블록과 해당 블록의 언두 데이터를 체인지 벡터 형태로 생성하여 이를 묶은 체인지 코드형태로 값을 저장하고 있는 버퍼이다. 이는 주로 512KB씩 8개의 블록으로 나누어서 관리된다.

 이 리두로그 버퍼를 자세하게 살펴보면 아래와 같다.

<img src="http://wiki.gurubee.net/download/attachments/28607002/6-2.png" />

 만약 임의의 세션이 리두로그버퍼에 있는 내용을 LGWR에 내려써야한다면, 'redo writing latch'를 획득하고 writing flag를 작성중으로 바꾼후, 'redo allocation latch'를 통해 프리포인터의 시작(포인터1)으로 이동한 후  프리포인터의 끝(포인터2)영역으로 이동하여 lgwr 프로세스로 리두 로그파일에 작성을 시작한다.



## 동작 원리

<img src="http://wiki.gurubee.net/download/attachments/28607002/6-3.png" />

  위에서 언급한 절차대로,

1. 'redo writing latch'를 통한 writting flag 변경
2. 'redo allocation latch'를 통한 프리 영역 시작 지점 catch & 끝 영역으로 이동
3. lgwr를 통한 로그파일 쓰기 작업 실시



※ piggyback commit

: 위의 예시에서 c1(첫번째 commit) 세션이 lgwr를 포스팅(호출)했다고 할 때, 위의 3가지 과정을 거치게 된다. 그런데, c2와 c3세션이 쓰기 작업이 발생하기 전에 commit이 완료된 상태라면, 호출을 c1 세션이 했지만, c2와 c3 세션에 대해서도 로그파일에 쓰기작업을 실행하는 것을 말한다.



 쓰기 작업이 끝났다면, 프리영역의 끝 포인터를 lgwr에 의해 기록이 끝난 영역까지 이동한다.

 만약, 임의의 리두가 리두 버퍼에 저장될 때는 어떤 프로세스를 따르게 될까?

1. redo copy latch를 획득
2. redo allocation latch를 통해 포인터의 시작지점을 찾음
3. redo allocation latch 릴리즈
4. 리두 copy 실시
5. redo copy latch 릴리즈

 이 때, 리두를 카피하면서 완료되지 않은 영역이 'alloc' 영역이다.

 redo copy가 일어나고 있는 동안에는, 리두 버퍼에 있는 내용을 lgwr를 통해 쓰기 작업을 중단한다. 그리하여, redo copy latch를 가지고 있는 세션을 확인하고, 만약 활성화된 세션이 있다면 'LGWR wait for redo copy' 대기이벤트를 발생하고 대기한다.



## 리두 블럭 낭비

<img src="http://wiki.gurubee.net/download/attachments/28607002/6-4.png" />

 이제 redo copy 작업을 마치고(블록에 alloc이 완료 된 후), c4 와 c5를 로그파일에 쓰기위해 대기하고 있던 lgwr가 포스팅되었다고 하자. 이때, 오라클은 94번 block에 남아있는 영역을 무시하고, 94번 전체 블럭에 대해서도 디스크에 기록한다.  효율적으로 관리해야 한다고 하는데, 왜 94번 블록의 끝으로 프리영역의 시작 포인터를 이동하여 기록하는 것일까?

 이는 디스크에 기록하는 블록에 대한 관리를 단순화하기 위해서이다.

 리두 로그 버퍼에 설정되어있는 블록은 디스크의 하나의 섹터 단위로 나누어져있다. 그리고 일반적인 기록작업은 디스크의 섹터 단위로 수행하게 된다. 그런데 프리영역의 시작부를 블록의 끝으로 이동하여 처리하지 않는다면, 다음 세션이 lgwr를 포스팅할 때, 이전에 사용했던 섹터를 불러와서 예전에 이미 기록해둔 블록까지 중복으로 쓰기작업을 할 수 있기 때문이다. 그래서 중복을 제거하기 위해 해당 섹터에서 이미 쓰여진 부분을 발라내는 작업이 필요하게 되고 이는 성능 문제를 야기한다.

 이렇게 이동을 통해 낭비되는 로그영역을 'redo wastage' 지표로 확인할 수 있다.