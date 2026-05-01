# SoftBlobGIN blob visualization for 2W0A
# Generated automatically — open in PyMOL

fetch 2w0a, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 21+38+39+50+52+74+137+158+160+162+235+236+296+302+329+330+362+408+413
color red, blob1
# Blob 1: 19 residues, importance=0.7656
select blob2, chain A and resi 24+26+40+41+70+77+80+128+129+163+164+209+227+233+249+292+315+412+415+419
color orange, blob2
# Blob 2: 20 residues, importance=0.0039
select blob3, chain A and resi 25+161+206+238+247+290+335+342+406+410+414
color yellow, blob3
# Blob 3: 11 residues, importance=0.0000
select blob4, chain A and resi 17+22+76+81+82+100+120+126+133+156+178+179+208+223+225+240+248+251+265+293+299+306+327+328+380+404+409+411
color green, blob4
# Blob 4: 28 residues, importance=0.0117
select blob5, chain A and resi 29+51+54+58+72+86+181+189+224+230+243+245+246+252+295+320+326+359+365+390
color cyan, blob5
# Blob 5: 20 residues, importance=0.0000
select blob6, chain A and resi 49+68+69+203+241+287+291+294+304+318+321+322+331+339+407
color blue, blob6
# Blob 6: 15 residues, importance=0.0117
select blob7, chain A and resi 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+18+19+23+27+28+31+32+33+34+35+36+37+43+44+45+46+47+53+55+56+57+59+60+62+63+65+71+75+78+83+84+85+87+88+89+90+91+92+93+94+95+96+98+102+103+104+105+106+107+108+109+110+111+112+113+114+115+116+117+119+121+122+125+127+130+131+132+135+136+138+139+140+142+143+144+145+146+147+148+149+151+153+154+157+165+166+167+168+169+170+171+172+173+174+176+180+182+183+184+185+186+187+188+192+193+194+195+196+197+198+199+200+201+202+204+205+210+211+212+213+214+215+216+217+218+219+221+228+229+232+234+244+250+253+254+255+256+258+260+261+264+266+267+268+269+270+271+272+273+275+276+277+278+279+280+281+282+283+284+285+289+298+300+301+307+309+310+312+313+314+316+324+332+333+337+338+340+341+343+344+345+346+347+348+349+350+351+352+353+354+355+356+357+360+361+363+364+366+367+368+369+370+371+372+373+374+375+376+378+381+382+383+384+385+386+387+389+391+392+393+394+396+397+398+399+400+401+402+416+417+423+424+425+426
color purple, blob7
# Blob 7: 250 residues, importance=0.1797
select blob8, chain A and resi 20+30+42+48+61+64+66+67+73+79+97+99+101+118+123+124+134+141+150+152+155+159+175+177+190+191+207+220+222+226+231+237+239+242+257+259+262+263+274+286+288+297+303+305+308+311+317+319+323+325+334+336+358+377+379+388+395+403+405+418+420+421+422
color white, blob8
# Blob 8: 63 residues, importance=0.0273

# Active/binding site residues
select active_site, chain A and resi 72+76+97+326+392+394
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 7-426

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A