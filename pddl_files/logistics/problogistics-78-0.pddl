(define (problem logistics-78-0)
(:domain logistics)
(:objects
    APT26
    POS26
    CIT26
    TRU26
    OBJ263
    OBJ262
    OBJ261
    APN7
    APT25
    POS25
    CIT25
    TRU25
    OBJ253
    OBJ252
    OBJ251
    APT24
    POS24
    CIT24
    TRU24
    OBJ243
    OBJ242
    OBJ241
    APT23
    POS23
    CIT23
    TRU23
    OBJ233
    OBJ232
    OBJ231
    APT22
    POS22
    CIT22
    TRU22
    OBJ223
    OBJ222
    OBJ221
    APN6
    APT21
    POS21
    CIT21
    TRU21
    OBJ213
    OBJ212
    OBJ211
    APT20
    POS20
    CIT20
    TRU20
    OBJ203
    OBJ202
    OBJ201
    APT19
    POS19
    CIT19
    TRU19
    OBJ193
    OBJ192
    OBJ191
    APT18
    POS18
    CIT18
    TRU18
    OBJ183
    OBJ182
    OBJ181
    APN5
    APT17
    POS17
    CIT17
    TRU17
    OBJ173
    OBJ172
    OBJ171
    APT16
    POS16
    CIT16
    TRU16
    OBJ163
    OBJ162
    OBJ161
    APT15
    POS15
    CIT15
    TRU15
    OBJ153
    OBJ152
    OBJ151
    APT14
    POS14
    CIT14
    TRU14
    OBJ143
    OBJ142
    OBJ141
    APN4
    APT13
    POS13
    CIT13
    TRU13
    OBJ133
    OBJ132
    OBJ131
    APT12
    POS12
    CIT12
    TRU12
    OBJ123
    OBJ122
    OBJ121
    APT11
    POS11
    CIT11
    TRU11
    OBJ113
    OBJ112
    OBJ111
    APT10
    POS10
    CIT10
    TRU10
    OBJ103
    OBJ102
    OBJ101
    APN3
    APT9
    POS9
    CIT9
    TRU9
    OBJ93
    OBJ92
    OBJ91
    APT8
    POS8
    CIT8
    TRU8
    OBJ83
    OBJ82
    OBJ81
    APT7
    POS7
    CIT7
    TRU7
    OBJ73
    OBJ72
    OBJ71
    APT6
    POS6
    CIT6
    TRU6
    OBJ63
    OBJ62
    OBJ61
    APN2
    APT5
    POS5
    CIT5
    TRU5
    OBJ53
    OBJ52
    OBJ51
    APT4
    POS4
    CIT4
    TRU4
    OBJ43
    OBJ42
    OBJ41
    APT3
    POS3
    CIT3
    TRU3
    OBJ33
    OBJ32
    OBJ31
    APT2
    POS2
    CIT2
    TRU2
    OBJ23
    OBJ22
    OBJ21
    APN1
    APT1
    POS1
    CIT1
    TRU1
    OBJ13
    OBJ12
    OBJ11
)
(:init
    (PACKAGE OBJ11)
    (PACKAGE OBJ12)
    (PACKAGE OBJ13)
    (PACKAGE OBJ21)
    (PACKAGE OBJ22)
    (PACKAGE OBJ23)
    (PACKAGE OBJ31)
    (PACKAGE OBJ32)
    (PACKAGE OBJ33)
    (PACKAGE OBJ41)
    (PACKAGE OBJ42)
    (PACKAGE OBJ43)
    (PACKAGE OBJ51)
    (PACKAGE OBJ52)
    (PACKAGE OBJ53)
    (PACKAGE OBJ61)
    (PACKAGE OBJ62)
    (PACKAGE OBJ63)
    (PACKAGE OBJ71)
    (PACKAGE OBJ72)
    (PACKAGE OBJ73)
    (PACKAGE OBJ81)
    (PACKAGE OBJ82)
    (PACKAGE OBJ83)
    (PACKAGE OBJ91)
    (PACKAGE OBJ92)
    (PACKAGE OBJ93)
    (PACKAGE OBJ101)
    (PACKAGE OBJ102)
    (PACKAGE OBJ103)
    (PACKAGE OBJ111)
    (PACKAGE OBJ112)
    (PACKAGE OBJ113)
    (PACKAGE OBJ121)
    (PACKAGE OBJ122)
    (PACKAGE OBJ123)
    (PACKAGE OBJ131)
    (PACKAGE OBJ132)
    (PACKAGE OBJ133)
    (PACKAGE OBJ141)
    (PACKAGE OBJ142)
    (PACKAGE OBJ143)
    (PACKAGE OBJ151)
    (PACKAGE OBJ152)
    (PACKAGE OBJ153)
    (PACKAGE OBJ161)
    (PACKAGE OBJ162)
    (PACKAGE OBJ163)
    (PACKAGE OBJ171)
    (PACKAGE OBJ172)
    (PACKAGE OBJ173)
    (PACKAGE OBJ181)
    (PACKAGE OBJ182)
    (PACKAGE OBJ183)
    (PACKAGE OBJ191)
    (PACKAGE OBJ192)
    (PACKAGE OBJ193)
    (PACKAGE OBJ201)
    (PACKAGE OBJ202)
    (PACKAGE OBJ203)
    (PACKAGE OBJ211)
    (PACKAGE OBJ212)
    (PACKAGE OBJ213)
    (PACKAGE OBJ221)
    (PACKAGE OBJ222)
    (PACKAGE OBJ223)
    (PACKAGE OBJ231)
    (PACKAGE OBJ232)
    (PACKAGE OBJ233)
    (PACKAGE OBJ241)
    (PACKAGE OBJ242)
    (PACKAGE OBJ243)
    (PACKAGE OBJ251)
    (PACKAGE OBJ252)
    (PACKAGE OBJ253)
    (PACKAGE OBJ261)
    (PACKAGE OBJ262)
    (PACKAGE OBJ263)
    (TRUCK TRU1)
    (TRUCK TRU2)
    (TRUCK TRU3)
    (TRUCK TRU4)
    (TRUCK TRU5)
    (TRUCK TRU6)
    (TRUCK TRU7)
    (TRUCK TRU8)
    (TRUCK TRU9)
    (TRUCK TRU10)
    (TRUCK TRU11)
    (TRUCK TRU12)
    (TRUCK TRU13)
    (TRUCK TRU14)
    (TRUCK TRU15)
    (TRUCK TRU16)
    (TRUCK TRU17)
    (TRUCK TRU18)
    (TRUCK TRU19)
    (TRUCK TRU20)
    (TRUCK TRU21)
    (TRUCK TRU22)
    (TRUCK TRU23)
    (TRUCK TRU24)
    (TRUCK TRU25)
    (TRUCK TRU26)
    (CITY CIT1)
    (CITY CIT2)
    (CITY CIT3)
    (CITY CIT4)
    (CITY CIT5)
    (CITY CIT6)
    (CITY CIT7)
    (CITY CIT8)
    (CITY CIT9)
    (CITY CIT10)
    (CITY CIT11)
    (CITY CIT12)
    (CITY CIT13)
    (CITY CIT14)
    (CITY CIT15)
    (CITY CIT16)
    (CITY CIT17)
    (CITY CIT18)
    (CITY CIT19)
    (CITY CIT20)
    (CITY CIT21)
    (CITY CIT22)
    (CITY CIT23)
    (CITY CIT24)
    (CITY CIT25)
    (CITY CIT26)
    (LOCATION POS1)
    (LOCATION APT1)
    (LOCATION POS2)
    (LOCATION APT2)
    (LOCATION POS3)
    (LOCATION APT3)
    (LOCATION POS4)
    (LOCATION APT4)
    (LOCATION POS5)
    (LOCATION APT5)
    (LOCATION POS6)
    (LOCATION APT6)
    (LOCATION POS7)
    (LOCATION APT7)
    (LOCATION POS8)
    (LOCATION APT8)
    (LOCATION POS9)
    (LOCATION APT9)
    (LOCATION POS10)
    (LOCATION APT10)
    (LOCATION POS11)
    (LOCATION APT11)
    (LOCATION POS12)
    (LOCATION APT12)
    (LOCATION POS13)
    (LOCATION APT13)
    (LOCATION POS14)
    (LOCATION APT14)
    (LOCATION POS15)
    (LOCATION APT15)
    (LOCATION POS16)
    (LOCATION APT16)
    (LOCATION POS17)
    (LOCATION APT17)
    (LOCATION POS18)
    (LOCATION APT18)
    (LOCATION POS19)
    (LOCATION APT19)
    (LOCATION POS20)
    (LOCATION APT20)
    (LOCATION POS21)
    (LOCATION APT21)
    (LOCATION POS22)
    (LOCATION APT22)
    (LOCATION POS23)
    (LOCATION APT23)
    (LOCATION POS24)
    (LOCATION APT24)
    (LOCATION POS25)
    (LOCATION APT25)
    (LOCATION POS26)
    (LOCATION APT26)
    (AIRPORT APT1)
    (AIRPORT APT2)
    (AIRPORT APT3)
    (AIRPORT APT4)
    (AIRPORT APT5)
    (AIRPORT APT6)
    (AIRPORT APT7)
    (AIRPORT APT8)
    (AIRPORT APT9)
    (AIRPORT APT10)
    (AIRPORT APT11)
    (AIRPORT APT12)
    (AIRPORT APT13)
    (AIRPORT APT14)
    (AIRPORT APT15)
    (AIRPORT APT16)
    (AIRPORT APT17)
    (AIRPORT APT18)
    (AIRPORT APT19)
    (AIRPORT APT20)
    (AIRPORT APT21)
    (AIRPORT APT22)
    (AIRPORT APT23)
    (AIRPORT APT24)
    (AIRPORT APT25)
    (AIRPORT APT26)
    (AIRPLANE APN1)
    (AIRPLANE APN2)
    (AIRPLANE APN3)
    (AIRPLANE APN4)
    (AIRPLANE APN5)
    (AIRPLANE APN6)
    (AIRPLANE APN7)
    (AT APN1 APT17)
    (AT APN2 APT13)
    (AT APN3 APT10)
    (AT APN4 APT16)
    (AT APN5 APT17)
    (AT APN6 APT24)
    (AT APN7 APT13)
    (AT TRU1 POS1)
    (AT OBJ11 POS1)
    (AT OBJ12 POS1)
    (AT OBJ13 POS1)
    (AT TRU2 POS2)
    (AT OBJ21 POS2)
    (AT OBJ22 POS2)
    (AT OBJ23 POS2)
    (AT TRU3 POS3)
    (AT OBJ31 POS3)
    (AT OBJ32 POS3)
    (AT OBJ33 POS3)
    (AT TRU4 POS4)
    (AT OBJ41 POS4)
    (AT OBJ42 POS4)
    (AT OBJ43 POS4)
    (AT TRU5 POS5)
    (AT OBJ51 POS5)
    (AT OBJ52 POS5)
    (AT OBJ53 POS5)
    (AT TRU6 POS6)
    (AT OBJ61 POS6)
    (AT OBJ62 POS6)
    (AT OBJ63 POS6)
    (AT TRU7 POS7)
    (AT OBJ71 POS7)
    (AT OBJ72 POS7)
    (AT OBJ73 POS7)
    (AT TRU8 POS8)
    (AT OBJ81 POS8)
    (AT OBJ82 POS8)
    (AT OBJ83 POS8)
    (AT TRU9 POS9)
    (AT OBJ91 POS9)
    (AT OBJ92 POS9)
    (AT OBJ93 POS9)
    (AT TRU10 POS10)
    (AT OBJ101 POS10)
    (AT OBJ102 POS10)
    (AT OBJ103 POS10)
    (AT TRU11 POS11)
    (AT OBJ111 POS11)
    (AT OBJ112 POS11)
    (AT OBJ113 POS11)
    (AT TRU12 POS12)
    (AT OBJ121 POS12)
    (AT OBJ122 POS12)
    (AT OBJ123 POS12)
    (AT TRU13 POS13)
    (AT OBJ131 POS13)
    (AT OBJ132 POS13)
    (AT OBJ133 POS13)
    (AT TRU14 POS14)
    (AT OBJ141 POS14)
    (AT OBJ142 POS14)
    (AT OBJ143 POS14)
    (AT TRU15 POS15)
    (AT OBJ151 POS15)
    (AT OBJ152 POS15)
    (AT OBJ153 POS15)
    (AT TRU16 POS16)
    (AT OBJ161 POS16)
    (AT OBJ162 POS16)
    (AT OBJ163 POS16)
    (AT TRU17 POS17)
    (AT OBJ171 POS17)
    (AT OBJ172 POS17)
    (AT OBJ173 POS17)
    (AT TRU18 POS18)
    (AT OBJ181 POS18)
    (AT OBJ182 POS18)
    (AT OBJ183 POS18)
    (AT TRU19 POS19)
    (AT OBJ191 POS19)
    (AT OBJ192 POS19)
    (AT OBJ193 POS19)
    (AT TRU20 POS20)
    (AT OBJ201 POS20)
    (AT OBJ202 POS20)
    (AT OBJ203 POS20)
    (AT TRU21 POS21)
    (AT OBJ211 POS21)
    (AT OBJ212 POS21)
    (AT OBJ213 POS21)
    (AT TRU22 POS22)
    (AT OBJ221 POS22)
    (AT OBJ222 POS22)
    (AT OBJ223 POS22)
    (AT TRU23 POS23)
    (AT OBJ231 POS23)
    (AT OBJ232 POS23)
    (AT OBJ233 POS23)
    (AT TRU24 POS24)
    (AT OBJ241 POS24)
    (AT OBJ242 POS24)
    (AT OBJ243 POS24)
    (AT TRU25 POS25)
    (AT OBJ251 POS25)
    (AT OBJ252 POS25)
    (AT OBJ253 POS25)
    (AT TRU26 POS26)
    (AT OBJ261 POS26)
    (AT OBJ262 POS26)
    (AT OBJ263 POS26)
    (IN-CITY POS1 CIT1)
    (IN-CITY APT1 CIT1)
    (IN-CITY POS2 CIT2)
    (IN-CITY APT2 CIT2)
    (IN-CITY POS3 CIT3)
    (IN-CITY APT3 CIT3)
    (IN-CITY POS4 CIT4)
    (IN-CITY APT4 CIT4)
    (IN-CITY POS5 CIT5)
    (IN-CITY APT5 CIT5)
    (IN-CITY POS6 CIT6)
    (IN-CITY APT6 CIT6)
    (IN-CITY POS7 CIT7)
    (IN-CITY APT7 CIT7)
    (IN-CITY POS8 CIT8)
    (IN-CITY APT8 CIT8)
    (IN-CITY POS9 CIT9)
    (IN-CITY APT9 CIT9)
    (IN-CITY POS10 CIT10)
    (IN-CITY APT10 CIT10)
    (IN-CITY POS11 CIT11)
    (IN-CITY APT11 CIT11)
    (IN-CITY POS12 CIT12)
    (IN-CITY APT12 CIT12)
    (IN-CITY POS13 CIT13)
    (IN-CITY APT13 CIT13)
    (IN-CITY POS14 CIT14)
    (IN-CITY APT14 CIT14)
    (IN-CITY POS15 CIT15)
    (IN-CITY APT15 CIT15)
    (IN-CITY POS16 CIT16)
    (IN-CITY APT16 CIT16)
    (IN-CITY POS17 CIT17)
    (IN-CITY APT17 CIT17)
    (IN-CITY POS18 CIT18)
    (IN-CITY APT18 CIT18)
    (IN-CITY POS19 CIT19)
    (IN-CITY APT19 CIT19)
    (IN-CITY POS20 CIT20)
    (IN-CITY APT20 CIT20)
    (IN-CITY POS21 CIT21)
    (IN-CITY APT21 CIT21)
    (IN-CITY POS22 CIT22)
    (IN-CITY APT22 CIT22)
    (IN-CITY POS23 CIT23)
    (IN-CITY APT23 CIT23)
    (IN-CITY POS24 CIT24)
    (IN-CITY APT24 CIT24)
    (IN-CITY POS25 CIT25)
    (IN-CITY APT25 CIT25)
    (IN-CITY POS26 CIT26)
    (IN-CITY APT26 CIT26)
)
(:goal (and
    (AT OBJ31 POS12)
    (AT OBJ241 POS4)
    (AT OBJ13 APT4)
    (AT OBJ201 APT19)
    (AT OBJ42 APT11)
    (AT OBJ12 POS25)
    (AT OBJ103 POS4)
    (AT OBJ162 APT18)
    (AT OBJ112 POS1)
    (AT OBJ43 POS14)
    (AT OBJ62 POS4)
    (AT OBJ253 APT3)
    (AT OBJ212 POS3)
    (AT OBJ163 APT5)
    (AT OBJ173 POS3)
    (AT OBJ171 APT18)
    (AT OBJ81 POS2)
    (AT OBJ51 APT17)
    (AT OBJ161 APT8)
    (AT OBJ53 APT10)
    (AT OBJ221 POS4)
    (AT OBJ111 POS18)
    (AT OBJ83 POS22)
    (AT OBJ123 POS16)
    (AT OBJ263 APT17)
    (AT OBJ91 APT8)
    (AT OBJ193 POS14)
    (AT OBJ243 POS15)
    (AT OBJ22 APT17)
    (AT OBJ92 APT21)
    (AT OBJ63 APT17)
    (AT OBJ142 APT26)
    (AT OBJ93 POS9)
    (AT OBJ143 POS9)
    (AT OBJ101 POS8)
    (AT OBJ252 POS6)
    (AT OBJ232 POS21)
    (AT OBJ113 APT16)
    (AT OBJ233 APT18)
    (AT OBJ72 APT25)
    (AT OBJ262 POS15)
    (AT OBJ211 POS8)
    (AT OBJ191 APT26)
    (AT OBJ33 POS21)
    (AT OBJ73 POS19)
    (AT OBJ182 POS6)
    (AT OBJ251 POS8)
    (AT OBJ242 APT2)
    (AT OBJ203 POS20)
    (AT OBJ122 APT3)
    (AT OBJ11 APT18)
    (AT OBJ71 APT24)
    (AT OBJ261 APT21)
    (AT OBJ41 POS1)
    (AT OBJ102 POS14)
    (AT OBJ23 POS18)
    (AT OBJ82 POS18)
    (AT OBJ141 POS23)
    (AT OBJ133 POS8)
    (AT OBJ153 APT9)
    (AT OBJ202 POS22)
    (AT OBJ192 POS21)
    (AT OBJ151 APT24)
    (AT OBJ172 APT22)
    (AT OBJ32 APT23)
    (AT OBJ152 APT4)
    (AT OBJ231 POS14)
    (AT OBJ121 APT15)
    (AT OBJ181 APT25)
    (AT OBJ213 POS4)
    (AT OBJ222 APT4)
    (AT OBJ61 APT18)
    (AT OBJ131 POS25)
    (AT OBJ21 APT10)
    (AT OBJ132 POS5)
    (AT OBJ223 APT5)
    (AT OBJ183 APT6)
    (AT OBJ52 APT14)
)
)
)
