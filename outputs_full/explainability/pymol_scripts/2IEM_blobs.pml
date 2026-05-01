# SoftBlobGIN blob visualization for 2IEM
# Generated automatically — open in PyMOL

fetch 2iem, async=0
remove not chain A
hide everything
show cartoon, chain A
color gray80, chain A

# Color residues by blob assignment
select blob1, chain A and resi 46+50+52+140+180+181+186+190
color red, blob1
# Blob 1: 8 residues, importance=0.6953
select blob2, chain A and resi 9+19+44+49+62+80+86+90+95+136+185+198
color orange, blob2
# Blob 2: 12 residues, importance=0.0039
select blob3, chain A and resi 54+73+89+93+96+137+177
color yellow, blob3
# Blob 3: 7 residues, importance=0.0000
select blob4, chain A and resi 15+45+60+85+99+134+135+138+146+149+150+178
color green, blob4
# Blob 4: 12 residues, importance=0.0117
select blob5, chain A and resi 55+70+75+79+100+133+179+182
color cyan, blob5
# Blob 5: 8 residues, importance=0.0039
select blob6, chain A and resi 47+51+71+76+81+189
color blue, blob6
# Blob 6: 6 residues, importance=0.0156
select blob7, chain A and resi 1+2+3+4+5+6+7+8+10+11+12+13+18+20+21+23+24+25+26+27+28+29+30+31+32+33+34+35+36+37+38+39+40+41+42+43+53+56+57+61+63+64+66+67+72+74+77+78+82+83+84+88+91+92+94+98+101+102+103+104+105+106+107+109+110+111+112+113+114+115+116+117+120+121+122+123+124+125+126+127+128+129+130+131+132+141+142+143+144+145+147+148+153+154+155+157+158+159+160+161+162+163+164+165+166+167+168+169+170+171+172+175+176+184+187+192+193+194+195+196+197+199+201+202+203+204+205+206+207+208+209+210+211
color purple, blob7
# Blob 7: 133 residues, importance=0.2109
select blob8, chain A and resi 14+16+17+22+48+58+59+65+68+69+87+97+108+118+119+139+151+152+156+173+174+183+188+191+200
color white, blob8
# Blob 8: 25 residues, importance=0.0586

# Active/binding site residues
select active_site, chain A and resi 51
show sticks, active_site
color magenta, active_site
set stick_radius, 0.15, active_site

# Domain boundaries
# Domain 0: residues 44-199

# Final styling
set cartoon_transparency, 0.3
set ray_shadow, 0
bg_color white
orient
zoom chain A