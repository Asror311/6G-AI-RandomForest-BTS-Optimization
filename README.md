# 6G AI Random Forest BTS Optimization

Ushbu loyiha 6G mobil aloqa tizimlarida foydalanuvchini optimal bazaviy stansiyaga ulash jarayonini Random Forest algoritmi yordamida modellashtirishga bag‘ishlangan.

## Loyihaning maqsadi

Modelning asosiy maqsadi foydalanuvchi uchun eng maqbul bazaviy stansiyani tanlash va tarmoq samaradorligini oshirishdan iborat. Bunda foydalanuvchi va bazaviy stansiyalar o‘rtasidagi masofa, signal kuchi, SINR, yuklama, kechikish va trafik hajmi kabi parametrlar hisobga olinadi.

## Foydalanilgan parametrlar

- Masofa
- Signal kuchi
- SINR
- Bazaviy stansiya yuklamasi
- Kechikish
- Trafik hajmi

## Taqqoslangan usullar

1. Eng yaqin bazaviy stansiyani tanlash
2. Eng kuchli signalga asoslangan tanlash
3. Random Forest AI modeli

## Asosiy natijalar

- Model aniqligi: 92.6%
- Random Forest AI samaradorligi: 77.45%
- Yuklama notekisligi: 0.03

## Ishga tushirish

```bash
pip install -r requirements.txt
python main.py
