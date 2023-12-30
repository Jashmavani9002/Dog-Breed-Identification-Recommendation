"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
# from django.conf.urls import urls
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home',views.index, name='index'),
    path('home_2',views.index_2, name='index_2'),
    path('home_3',views.index_3, name='index_3'),
    path('about',views.about, name='about'),
    path('login',views.login, name='login'),
    path('',views.loggedout, name='loggedout'),
    path('signup',views.signup, name='signup'),
    path('logout',views.logout,name='logout'),
    path('back_button',views.back, name='back'),
    path('recommendation_1',views.setrecommend,name='recommendation'),
    path('recommend_1',views.recommend, name='recommend'),
    path('contact',views.contact, name='contact'),
    path('img_submit',views.geturl, name='geturl'),
    path('myphotos_2',views.myphotos, name='myphotos'),
    path('myprofile_2',views.myprofile, name='myprofile'),
    path('mydp_2',views.mydp, name='mydp'),
    path('dp_submit',views.setdp,name='setdp'),
    path('classi_2',views.classi,name = 'classi'),
    path('classi_img',views.setclassi,name='setclassi')

]
