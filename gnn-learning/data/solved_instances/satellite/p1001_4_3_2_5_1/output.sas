begin_version
3
end_version
begin_metric
0
end_metric
34
begin_variable
var0
-1
2
Atom calibrated(instrument0)
NegatedAtom calibrated(instrument0)
end_variable
begin_variable
var1
-1
2
Atom calibrated(instrument1)
NegatedAtom calibrated(instrument1)
end_variable
begin_variable
var2
-1
2
Atom calibrated(instrument2)
NegatedAtom calibrated(instrument2)
end_variable
begin_variable
var3
-1
2
Atom calibrated(instrument3)
NegatedAtom calibrated(instrument3)
end_variable
begin_variable
var4
-1
2
Atom calibrated(instrument4)
NegatedAtom calibrated(instrument4)
end_variable
begin_variable
var5
-1
2
Atom calibrated(instrument5)
NegatedAtom calibrated(instrument5)
end_variable
begin_variable
var6
-1
2
Atom calibrated(instrument6)
NegatedAtom calibrated(instrument6)
end_variable
begin_variable
var7
-1
2
Atom have_image(groundstation0, image1)
NegatedAtom have_image(groundstation0, image1)
end_variable
begin_variable
var8
-1
2
Atom have_image(groundstation0, thermograph0)
NegatedAtom have_image(groundstation0, thermograph0)
end_variable
begin_variable
var9
-1
2
Atom have_image(groundstation2, image1)
NegatedAtom have_image(groundstation2, image1)
end_variable
begin_variable
var10
-1
2
Atom have_image(groundstation2, thermograph0)
NegatedAtom have_image(groundstation2, thermograph0)
end_variable
begin_variable
var11
-1
2
Atom have_image(groundstation3, image1)
NegatedAtom have_image(groundstation3, image1)
end_variable
begin_variable
var12
-1
2
Atom have_image(groundstation3, thermograph0)
NegatedAtom have_image(groundstation3, thermograph0)
end_variable
begin_variable
var13
-1
2
Atom have_image(phenomenon5, image1)
NegatedAtom have_image(phenomenon5, image1)
end_variable
begin_variable
var14
-1
2
Atom have_image(phenomenon5, thermograph0)
NegatedAtom have_image(phenomenon5, thermograph0)
end_variable
begin_variable
var15
-1
2
Atom have_image(star1, image1)
NegatedAtom have_image(star1, image1)
end_variable
begin_variable
var16
-1
2
Atom have_image(star1, thermograph0)
NegatedAtom have_image(star1, thermograph0)
end_variable
begin_variable
var17
-1
2
Atom have_image(star4, image1)
NegatedAtom have_image(star4, image1)
end_variable
begin_variable
var18
-1
2
Atom have_image(star4, thermograph0)
NegatedAtom have_image(star4, thermograph0)
end_variable
begin_variable
var19
-1
6
Atom pointing(satellite0, groundstation0)
Atom pointing(satellite0, groundstation2)
Atom pointing(satellite0, groundstation3)
Atom pointing(satellite0, phenomenon5)
Atom pointing(satellite0, star1)
Atom pointing(satellite0, star4)
end_variable
begin_variable
var20
-1
6
Atom pointing(satellite1, groundstation0)
Atom pointing(satellite1, groundstation2)
Atom pointing(satellite1, groundstation3)
Atom pointing(satellite1, phenomenon5)
Atom pointing(satellite1, star1)
Atom pointing(satellite1, star4)
end_variable
begin_variable
var21
-1
6
Atom pointing(satellite2, groundstation0)
Atom pointing(satellite2, groundstation2)
Atom pointing(satellite2, groundstation3)
Atom pointing(satellite2, phenomenon5)
Atom pointing(satellite2, star1)
Atom pointing(satellite2, star4)
end_variable
begin_variable
var22
-1
6
Atom pointing(satellite3, groundstation0)
Atom pointing(satellite3, groundstation2)
Atom pointing(satellite3, groundstation3)
Atom pointing(satellite3, phenomenon5)
Atom pointing(satellite3, star1)
Atom pointing(satellite3, star4)
end_variable
begin_variable
var23
-1
2
Atom power_avail(satellite0)
NegatedAtom power_avail(satellite0)
end_variable
begin_variable
var24
-1
2
Atom power_avail(satellite1)
NegatedAtom power_avail(satellite1)
end_variable
begin_variable
var25
-1
2
Atom power_avail(satellite2)
NegatedAtom power_avail(satellite2)
end_variable
begin_variable
var26
-1
2
Atom power_avail(satellite3)
NegatedAtom power_avail(satellite3)
end_variable
begin_variable
var27
-1
2
Atom power_on(instrument0)
NegatedAtom power_on(instrument0)
end_variable
begin_variable
var28
-1
2
Atom power_on(instrument1)
NegatedAtom power_on(instrument1)
end_variable
begin_variable
var29
-1
2
Atom power_on(instrument2)
NegatedAtom power_on(instrument2)
end_variable
begin_variable
var30
-1
2
Atom power_on(instrument3)
NegatedAtom power_on(instrument3)
end_variable
begin_variable
var31
-1
2
Atom power_on(instrument4)
NegatedAtom power_on(instrument4)
end_variable
begin_variable
var32
-1
2
Atom power_on(instrument5)
NegatedAtom power_on(instrument5)
end_variable
begin_variable
var33
-1
2
Atom power_on(instrument6)
NegatedAtom power_on(instrument6)
end_variable
4
begin_mutex_group
6
19 0
19 1
19 2
19 3
19 4
19 5
end_mutex_group
begin_mutex_group
6
20 0
20 1
20 2
20 3
20 4
20 5
end_mutex_group
begin_mutex_group
6
21 0
21 1
21 2
21 3
21 4
21 5
end_mutex_group
begin_mutex_group
6
22 0
22 1
22 2
22 3
22 4
22 5
end_mutex_group
begin_state
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
2
0
2
4
0
0
0
0
1
1
1
1
1
1
1
end_state
begin_goal
3
13 0
20 1
22 5
end_goal
213
begin_operator
calibrate satellite0 instrument0 groundstation0
2
19 0
27 0
1
0 0 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument1 star4
2
19 5
28 0
1
0 1 -1 0
1
end_operator
begin_operator
calibrate satellite0 instrument2 groundstation3
2
19 2
29 0
1
0 2 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument3 star4
2
20 5
30 0
1
0 3 -1 0
1
end_operator
begin_operator
calibrate satellite1 instrument4 star4
2
20 5
31 0
1
0 4 -1 0
1
end_operator
begin_operator
calibrate satellite2 instrument5 groundstation0
2
21 0
32 0
1
0 5 -1 0
1
end_operator
begin_operator
calibrate satellite3 instrument6 groundstation2
2
22 1
33 0
1
0 6 -1 0
1
end_operator
begin_operator
switch_off instrument0 satellite0
0
2
0 23 -1 0
0 27 0 1
1
end_operator
begin_operator
switch_off instrument1 satellite0
0
2
0 23 -1 0
0 28 0 1
1
end_operator
begin_operator
switch_off instrument2 satellite0
0
2
0 23 -1 0
0 29 0 1
1
end_operator
begin_operator
switch_off instrument3 satellite1
0
2
0 24 -1 0
0 30 0 1
1
end_operator
begin_operator
switch_off instrument4 satellite1
0
2
0 24 -1 0
0 31 0 1
1
end_operator
begin_operator
switch_off instrument5 satellite2
0
2
0 25 -1 0
0 32 0 1
1
end_operator
begin_operator
switch_off instrument6 satellite3
0
2
0 26 -1 0
0 33 0 1
1
end_operator
begin_operator
switch_on instrument0 satellite0
0
3
0 0 -1 1
0 23 0 1
0 27 -1 0
1
end_operator
begin_operator
switch_on instrument1 satellite0
0
3
0 1 -1 1
0 23 0 1
0 28 -1 0
1
end_operator
begin_operator
switch_on instrument2 satellite0
0
3
0 2 -1 1
0 23 0 1
0 29 -1 0
1
end_operator
begin_operator
switch_on instrument3 satellite1
0
3
0 3 -1 1
0 24 0 1
0 30 -1 0
1
end_operator
begin_operator
switch_on instrument4 satellite1
0
3
0 4 -1 1
0 24 0 1
0 31 -1 0
1
end_operator
begin_operator
switch_on instrument5 satellite2
0
3
0 5 -1 1
0 25 0 1
0 32 -1 0
1
end_operator
begin_operator
switch_on instrument6 satellite3
0
3
0 6 -1 1
0 26 0 1
0 33 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument0 thermograph0
3
0 0
19 0
27 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument1 image1
3
1 0
19 0
28 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument1 thermograph0
3
1 0
19 0
28 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument2 image1
3
2 0
19 0
29 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation0 instrument2 thermograph0
3
2 0
19 0
29 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument0 thermograph0
3
0 0
19 1
27 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument1 image1
3
1 0
19 1
28 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument1 thermograph0
3
1 0
19 1
28 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument2 image1
3
2 0
19 1
29 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation2 instrument2 thermograph0
3
2 0
19 1
29 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument0 thermograph0
3
0 0
19 2
27 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument1 image1
3
1 0
19 2
28 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument1 thermograph0
3
1 0
19 2
28 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument2 image1
3
2 0
19 2
29 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite0 groundstation3 instrument2 thermograph0
3
2 0
19 2
29 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon5 instrument0 thermograph0
3
0 0
19 3
27 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon5 instrument1 image1
3
1 0
19 3
28 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon5 instrument1 thermograph0
3
1 0
19 3
28 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon5 instrument2 image1
3
2 0
19 3
29 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite0 phenomenon5 instrument2 thermograph0
3
2 0
19 3
29 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite0 star1 instrument0 thermograph0
3
0 0
19 4
27 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star1 instrument1 image1
3
1 0
19 4
28 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star1 instrument1 thermograph0
3
1 0
19 4
28 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star1 instrument2 image1
3
2 0
19 4
29 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite0 star1 instrument2 thermograph0
3
2 0
19 4
29 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument0 thermograph0
3
0 0
19 5
27 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument1 image1
3
1 0
19 5
28 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument1 thermograph0
3
1 0
19 5
28 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument2 image1
3
2 0
19 5
29 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite0 star4 instrument2 thermograph0
3
2 0
19 5
29 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument3 image1
3
3 0
20 0
30 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument3 thermograph0
3
3 0
20 0
30 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument4 image1
3
4 0
20 0
31 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation0 instrument4 thermograph0
3
4 0
20 0
31 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument3 image1
3
3 0
20 1
30 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument3 thermograph0
3
3 0
20 1
30 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument4 image1
3
4 0
20 1
31 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation2 instrument4 thermograph0
3
4 0
20 1
31 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument3 image1
3
3 0
20 2
30 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument3 thermograph0
3
3 0
20 2
30 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument4 image1
3
4 0
20 2
31 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite1 groundstation3 instrument4 thermograph0
3
4 0
20 2
31 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon5 instrument3 image1
3
3 0
20 3
30 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon5 instrument3 thermograph0
3
3 0
20 3
30 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon5 instrument4 image1
3
4 0
20 3
31 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite1 phenomenon5 instrument4 thermograph0
3
4 0
20 3
31 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite1 star1 instrument3 image1
3
3 0
20 4
30 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 star1 instrument3 thermograph0
3
3 0
20 4
30 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 star1 instrument4 image1
3
4 0
20 4
31 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite1 star1 instrument4 thermograph0
3
4 0
20 4
31 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument3 image1
3
3 0
20 5
30 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument3 thermograph0
3
3 0
20 5
30 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument4 image1
3
4 0
20 5
31 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite1 star4 instrument4 thermograph0
3
4 0
20 5
31 0
1
0 18 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation0 instrument5 image1
3
5 0
21 0
32 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation2 instrument5 image1
3
5 0
21 1
32 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite2 groundstation3 instrument5 image1
3
5 0
21 2
32 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite2 phenomenon5 instrument5 image1
3
5 0
21 3
32 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite2 star1 instrument5 image1
3
5 0
21 4
32 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite2 star4 instrument5 image1
3
5 0
21 5
32 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation0 instrument6 image1
3
6 0
22 0
33 0
1
0 7 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation0 instrument6 thermograph0
3
6 0
22 0
33 0
1
0 8 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation2 instrument6 image1
3
6 0
22 1
33 0
1
0 9 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation2 instrument6 thermograph0
3
6 0
22 1
33 0
1
0 10 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation3 instrument6 image1
3
6 0
22 2
33 0
1
0 11 -1 0
1
end_operator
begin_operator
take_image satellite3 groundstation3 instrument6 thermograph0
3
6 0
22 2
33 0
1
0 12 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon5 instrument6 image1
3
6 0
22 3
33 0
1
0 13 -1 0
1
end_operator
begin_operator
take_image satellite3 phenomenon5 instrument6 thermograph0
3
6 0
22 3
33 0
1
0 14 -1 0
1
end_operator
begin_operator
take_image satellite3 star1 instrument6 image1
3
6 0
22 4
33 0
1
0 15 -1 0
1
end_operator
begin_operator
take_image satellite3 star1 instrument6 thermograph0
3
6 0
22 4
33 0
1
0 16 -1 0
1
end_operator
begin_operator
take_image satellite3 star4 instrument6 image1
3
6 0
22 5
33 0
1
0 17 -1 0
1
end_operator
begin_operator
take_image satellite3 star4 instrument6 thermograph0
3
6 0
22 5
33 0
1
0 18 -1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation2
0
1
0 19 1 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 groundstation3
0
1
0 19 2 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 phenomenon5
0
1
0 19 3 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star1
0
1
0 19 4 0
1
end_operator
begin_operator
turn_to satellite0 groundstation0 star4
0
1
0 19 5 0
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation0
0
1
0 19 0 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 groundstation3
0
1
0 19 2 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 phenomenon5
0
1
0 19 3 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star1
0
1
0 19 4 1
1
end_operator
begin_operator
turn_to satellite0 groundstation2 star4
0
1
0 19 5 1
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation0
0
1
0 19 0 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 groundstation2
0
1
0 19 1 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 phenomenon5
0
1
0 19 3 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star1
0
1
0 19 4 2
1
end_operator
begin_operator
turn_to satellite0 groundstation3 star4
0
1
0 19 5 2
1
end_operator
begin_operator
turn_to satellite0 phenomenon5 groundstation0
0
1
0 19 0 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon5 groundstation2
0
1
0 19 1 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon5 groundstation3
0
1
0 19 2 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon5 star1
0
1
0 19 4 3
1
end_operator
begin_operator
turn_to satellite0 phenomenon5 star4
0
1
0 19 5 3
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation0
0
1
0 19 0 4
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation2
0
1
0 19 1 4
1
end_operator
begin_operator
turn_to satellite0 star1 groundstation3
0
1
0 19 2 4
1
end_operator
begin_operator
turn_to satellite0 star1 phenomenon5
0
1
0 19 3 4
1
end_operator
begin_operator
turn_to satellite0 star1 star4
0
1
0 19 5 4
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation0
0
1
0 19 0 5
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation2
0
1
0 19 1 5
1
end_operator
begin_operator
turn_to satellite0 star4 groundstation3
0
1
0 19 2 5
1
end_operator
begin_operator
turn_to satellite0 star4 phenomenon5
0
1
0 19 3 5
1
end_operator
begin_operator
turn_to satellite0 star4 star1
0
1
0 19 4 5
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation2
0
1
0 20 1 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 groundstation3
0
1
0 20 2 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 phenomenon5
0
1
0 20 3 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star1
0
1
0 20 4 0
1
end_operator
begin_operator
turn_to satellite1 groundstation0 star4
0
1
0 20 5 0
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation0
0
1
0 20 0 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 groundstation3
0
1
0 20 2 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 phenomenon5
0
1
0 20 3 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star1
0
1
0 20 4 1
1
end_operator
begin_operator
turn_to satellite1 groundstation2 star4
0
1
0 20 5 1
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation0
0
1
0 20 0 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 groundstation2
0
1
0 20 1 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 phenomenon5
0
1
0 20 3 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star1
0
1
0 20 4 2
1
end_operator
begin_operator
turn_to satellite1 groundstation3 star4
0
1
0 20 5 2
1
end_operator
begin_operator
turn_to satellite1 phenomenon5 groundstation0
0
1
0 20 0 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon5 groundstation2
0
1
0 20 1 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon5 groundstation3
0
1
0 20 2 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon5 star1
0
1
0 20 4 3
1
end_operator
begin_operator
turn_to satellite1 phenomenon5 star4
0
1
0 20 5 3
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation0
0
1
0 20 0 4
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation2
0
1
0 20 1 4
1
end_operator
begin_operator
turn_to satellite1 star1 groundstation3
0
1
0 20 2 4
1
end_operator
begin_operator
turn_to satellite1 star1 phenomenon5
0
1
0 20 3 4
1
end_operator
begin_operator
turn_to satellite1 star1 star4
0
1
0 20 5 4
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation0
0
1
0 20 0 5
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation2
0
1
0 20 1 5
1
end_operator
begin_operator
turn_to satellite1 star4 groundstation3
0
1
0 20 2 5
1
end_operator
begin_operator
turn_to satellite1 star4 phenomenon5
0
1
0 20 3 5
1
end_operator
begin_operator
turn_to satellite1 star4 star1
0
1
0 20 4 5
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation2
0
1
0 21 1 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 groundstation3
0
1
0 21 2 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 phenomenon5
0
1
0 21 3 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star1
0
1
0 21 4 0
1
end_operator
begin_operator
turn_to satellite2 groundstation0 star4
0
1
0 21 5 0
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation0
0
1
0 21 0 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 groundstation3
0
1
0 21 2 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 phenomenon5
0
1
0 21 3 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star1
0
1
0 21 4 1
1
end_operator
begin_operator
turn_to satellite2 groundstation2 star4
0
1
0 21 5 1
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation0
0
1
0 21 0 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 groundstation2
0
1
0 21 1 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 phenomenon5
0
1
0 21 3 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star1
0
1
0 21 4 2
1
end_operator
begin_operator
turn_to satellite2 groundstation3 star4
0
1
0 21 5 2
1
end_operator
begin_operator
turn_to satellite2 phenomenon5 groundstation0
0
1
0 21 0 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon5 groundstation2
0
1
0 21 1 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon5 groundstation3
0
1
0 21 2 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon5 star1
0
1
0 21 4 3
1
end_operator
begin_operator
turn_to satellite2 phenomenon5 star4
0
1
0 21 5 3
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation0
0
1
0 21 0 4
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation2
0
1
0 21 1 4
1
end_operator
begin_operator
turn_to satellite2 star1 groundstation3
0
1
0 21 2 4
1
end_operator
begin_operator
turn_to satellite2 star1 phenomenon5
0
1
0 21 3 4
1
end_operator
begin_operator
turn_to satellite2 star1 star4
0
1
0 21 5 4
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation0
0
1
0 21 0 5
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation2
0
1
0 21 1 5
1
end_operator
begin_operator
turn_to satellite2 star4 groundstation3
0
1
0 21 2 5
1
end_operator
begin_operator
turn_to satellite2 star4 phenomenon5
0
1
0 21 3 5
1
end_operator
begin_operator
turn_to satellite2 star4 star1
0
1
0 21 4 5
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation2
0
1
0 22 1 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 groundstation3
0
1
0 22 2 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 phenomenon5
0
1
0 22 3 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star1
0
1
0 22 4 0
1
end_operator
begin_operator
turn_to satellite3 groundstation0 star4
0
1
0 22 5 0
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation0
0
1
0 22 0 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 groundstation3
0
1
0 22 2 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 phenomenon5
0
1
0 22 3 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star1
0
1
0 22 4 1
1
end_operator
begin_operator
turn_to satellite3 groundstation2 star4
0
1
0 22 5 1
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation0
0
1
0 22 0 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 groundstation2
0
1
0 22 1 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 phenomenon5
0
1
0 22 3 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star1
0
1
0 22 4 2
1
end_operator
begin_operator
turn_to satellite3 groundstation3 star4
0
1
0 22 5 2
1
end_operator
begin_operator
turn_to satellite3 phenomenon5 groundstation0
0
1
0 22 0 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon5 groundstation2
0
1
0 22 1 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon5 groundstation3
0
1
0 22 2 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon5 star1
0
1
0 22 4 3
1
end_operator
begin_operator
turn_to satellite3 phenomenon5 star4
0
1
0 22 5 3
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation0
0
1
0 22 0 4
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation2
0
1
0 22 1 4
1
end_operator
begin_operator
turn_to satellite3 star1 groundstation3
0
1
0 22 2 4
1
end_operator
begin_operator
turn_to satellite3 star1 phenomenon5
0
1
0 22 3 4
1
end_operator
begin_operator
turn_to satellite3 star1 star4
0
1
0 22 5 4
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation0
0
1
0 22 0 5
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation2
0
1
0 22 1 5
1
end_operator
begin_operator
turn_to satellite3 star4 groundstation3
0
1
0 22 2 5
1
end_operator
begin_operator
turn_to satellite3 star4 phenomenon5
0
1
0 22 3 5
1
end_operator
begin_operator
turn_to satellite3 star4 star1
0
1
0 22 4 5
1
end_operator
0
