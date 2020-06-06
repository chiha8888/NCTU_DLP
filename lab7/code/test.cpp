//
// Created by dorawu on 2020/6/4.
//
#include <bits/stdc++.h>
#include <iostream>
#include <fstream>

float A[10]={100,200,300,400,500,600,700,800,900,1000};

int main(){
    std::ofstream file;
    file.open("../score.csv");
    file<<"score\n";
    for (int i = 0; i < sizeof(A)/sizeof(A[0]); ++i) {
        file<<A[i]<<'\n';
    }
    file.close();

    return 0;
}