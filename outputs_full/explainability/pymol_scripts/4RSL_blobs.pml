# SoftBlobGIN blob visualization for 4RSL
# Generated automatically — open in PyMOL

fetch 4rsl, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 51+95+97+223+234+235+243+269+286+309+318+334+352+376+377
color red, blob1
# Blob 1: 15 residues, importance=0.7695
select blob2, chain A and resi 57+154+190+230+245+248+277+306+311+323+346+373+385+418+424+428
color orange, blob2
# Blob 2: 16 residues, importance=0.0000
select blob3, chain A and resi 91+99+237+260+261+263+266+290+291+296+307+329+331+350+354+375+427
color yellow, blob3
# Blob 3: 17 residues, importance=0.0000
select blob4, chain A and resi 11+18+22+39+50+56+88+93+96+98+103+107+127+150+155+159+161+163+169+170+215+227+236+238+253+256+283+292+297+320+345+347+349+357+382+386+404+415+423
color green, blob4
# Blob 4: 39 residues, importance=0.0117
select blob5, chain A and resi 21+92+219+241+281+285+304+326+330+417
color cyan, blob5
# Blob 5: 10 residues, importance=0.0039
select blob6, chain A and resi 16+153+156+222+231+246+247+264+273+319+332+343+425
color blue, blob6
# Blob 6: 13 residues, importance=0.0156
select blob7, chain A and resi 1+2+3+4+5+6+7+8+9+12+14+15+19+20+23+25+26+27+28+29+30+31+32+33+34+35+36+37+38+41+43+44+45+46+48+49+52+55+58+59+61+62+63+64+65+66+67+68+69+70+71+72+73+74+75+76+77+78+79+80+81+82+83+84+85+86+87+90+100+101+102+104+105+106+108+109+110+111+112+113+114+115+116+117+118+119+120+121+122+123+124+125+126+128+129+130+132+133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148+149+152+162+164+165+167+168+171+172+173+175+176+177+179+180+182+183+184+187+188+189+191+192+193+194+195+196+197+198+199+200+201+202+203+204+205+206+207+208+209+210+211+212+213+214+216+217+218+220+226+228+229+232+233+239+242+249+250+251+252+254+255+257+258+259+262+265+268+270+271+272+275+276+278+279+280+284+287+294+298+299+300+301+302+303+308+310+312+313+314+315+316+321+322+325+327+328+335+336+337+338+339+340+341+348+351+356+358+361+362+363+364+365+366+367+368+369+370+371+372+378+379+380+384+387+389+390+391+392+393+394+395+396+397+398+399+400+401+402+403+405+406+407+408+409+410+412+413+416+419+421+422+429+430+431+433+434+435+436+437+438+439
color purple, blob7
# Blob 7: 275 residues, importance=0.1719
select blob8, chain A and resi 10+13+17+24+40+42+47+53+54+60+89+94+131+151+157+158+160+166+174+178+181+185+186+221+224+225+240+244+267+274+282+288+289+293+295+305+317+324+333+342+344+353+355+359+360+374+381+383+388+411+414+420+426+432
color white, blob8
# Blob 8: 54 residues, importance=0.0273

# Active/binding site residues
select active_site, chain A and resi 18+19+41+47+49+50+51+52+58+188+374+378+379+380
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 2-90
# Domain 1: residues 157-234
# Domain 2: residues 353-411
# Domain 3: residues 91-156
# Domain 4: residues 235-352
# Domain 5: residues 412-437

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A