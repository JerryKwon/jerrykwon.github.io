---
layout: post
title: "[개발환경 구축] Centos7 VirtualBox Guest Addition 설치"
subtitle: Centos7 VirtualBox Guest Addition 설치
categories: database
date: 2020-04-08T21:15:00+09:00
tags: infra virtualbox centos7
comments: true
---
## 개요
> Oracle Virtual Box 가상머신에서 파일을 up/download시 사용되는 guest addition 설치시 발생 에러를 조치하는 내용입니다.

## VirtualBox Guest Addition

일단 이를 말하기에 앞서, VirtualBox는 Oracle 사에서 제공하는 가상화된 서버를 만들수 있게 도와주는 도구이다.

 간단히 말해, Mac의 경우 Window OS를 별도로 설치하고 OS를 이중화하여 사용하는 것과 비슷하다.

(사실 Mac을 사용하지 않아서 정확한 명칭도 잘 모르겠으나, 운영체제 위에 또 다른 운영체제임을 강조하고 싶었다.)



임의의 환경을 테스트하는데 있어서 가상화된 환경을 사용하면 좋은 장점은, 서버에 테스트하고 수정/삭제하는데 굉장히 용이한 것 같다. 다른 장점에 대해서는 직접 알아 보길 바란다.



Guest Addition은 VirtualBox로 기동된 가상화 환경에 Parent OS의 파일을 드롭다운 식으로 전달하거나, 화면 해상도 자동조절과 같은 목적을 위해 사용하게 될 것이다.



VB Guest Addition을 설치하기 위한 절차는 아래와 같다.



1.  VB에 가상화한 서버를 실행

2. [장치] - 게스트 확장 CD 이미지 삽입

3. 아래의 커맨드 입력

   ```
   yum groupinstall "Development Tools"
   yum install kernel-devel
   ```

   (만약 다른 언어를 채택하고 있다면, yum grouplist로 "Development Tools"에 해당하는 문자열로 대체)

4. Mount된 Disk에 설치파일 실행

5. Disk Unmount

6. 재부팅



위의 절차 이후, 가상화 서버의 창을 키워서 해상도가 자동으로 조절되는 지 확인해본다.



출처

http://egloos.zum.com/dochi575/v/4835884

https://wiki.centos.org/HowTos/Virtualization/VirtualBox/CentOSguest