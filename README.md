# 内容
本コードは、再構築した密度場のパワースペクトルの1ループオーダーの摂動計算を行うコードです。

現在観測される銀河やダークマターの空間分布の構造(以下、宇宙大規模構造)は、重力によって物質が集まって形成されたものです。
重力による成長は非線形なプロセスであるため、ゆらぎの摂動論は非線形性の強い小スケールにいくほど適用できなくなります。

大スケールの速度場(バルクモーション)の影響を補正し、物質素片を初期位置付近に戻すことで、
線形成長に近い密度場を再構築することができます。

本コードによって、再構築した密度場においてパワースペクトルの１ループオーダーの摂動成分を計算した結果、
より小スケールまで摂動論が適用できることが分かりました。
また構造の成長率を表すパラメターの決定精度が改善することも確かめました。

密度場を再構築するコードは以下をご覧ください
https://github.com/chiaki-hikage/reconstruct_densityfield

# Referernces

- Perturbation Theory for BAO reconstructed fields: one-loop results in real-space matter density field  
Chiaki Hikage, Kazuya Koyama, Alan Heavens  
Phys. Rev. D, Vol. 96 (2017) id.043513

- Perturbation theory for the redshift-space matter power spectra after reconstruction  
Chiaki Hikage, Kazuya Koyama, Ryuichi Takahashi  
Phys. Rev. D, Vol. 101, Issue 4 (2020), id.043510
