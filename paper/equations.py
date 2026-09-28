# -*- coding: utf-8 -*-
"""LaTeX source for every display equation, keyed by its number.

The builder converts these to native Word equation objects (OMML) via
latex2mathml -> MathML -> Word's own MML2OMML.XSL. Editing an equation here
changes the real Word equation, not a text imitation of one.
"""

DISPLAY = {
 1: r"O_j = \left( \hat{I}_j ,\; \hat{Z}_j ,\; V_j ,\; Q_j ,\; U_j \right)",
 # j indexes a camera here, as in Equation 1; a superscript t marks a
 # target-side quantity. Writing the held-out camera as a subscript t made the
 # same symbol an index in these two equations and a label in Equation 4.
 2: r"\hat{I}_j = F_\theta\left( \{ I_i, K_i, D_i, T_i \}_{i \neq j} ,\; P,\; K_j, D_j, T_j \right)",
 3: r"\hat{I}'_j = F_\theta\left( \{ I_i, K_i, D_i, T_i \}_{i=1}^{7} ,\; P,\; K'_j, D'_j, T'_j \right)",
 4: r"\hat{I}_j^{\mathrm{L4}} = F_\theta\left( \{ I_i^{\mathrm{src}}, K_i^{\mathrm{src}}, D_i^{\mathrm{src}}, T_i^{\mathrm{src}} \} ,\; P,\; K_j^{\mathrm{L4}}, D_j^{\mathrm{L4}}, T_j^{\mathrm{L4}} \right)",
 5: r"T_{L \leftarrow C_j}^{\mathrm{view}} = T_{L \leftarrow R}^{\mathrm{view}} \cdot T_{R \leftarrow C_j}^{\mathrm{mount}}",
 6: r"t' = t - R\,\Delta C",
 7: r"Q(p) \in \{ \mathrm{observed},\; \mathrm{gate},\; \mathrm{generated} \}",
 8: r"I^{\mathrm{rect}} = \mathrm{Remap}\left( I^{\mathrm{raw}},\; K, D \rightarrow K', 0 \right)",
 9: r"\tilde{p}_i = K'_i\, T_i\, X , \qquad p_i = \left( \tilde{p}_{i,x} / \tilde{p}_{i,z} ,\; \tilde{p}_{i,y} / \tilde{p}_{i,z} \right)",
10: r"d^*(p) = d_{\mathrm{DA3}}(p) \cdot \exp\left( g(p) \cdot r_\theta(p) \right)",
11: r"q_k = \frac{1}{d_{\max}} + \frac{k}{N_d-1}\left( \frac{1}{d_{\min}} - \frac{1}{d_{\max}} \right) , \qquad d_k = \frac{1}{q_k}",
12: r"\mu_k(p) = \frac{1}{N_k} \sum_i M_{i,k}(p)\, F_{i,k}(p)",
13: r"V_k(p) = \frac{1}{N_k} \sum_i M_{i,k}(p)\, F_{i,k}^{2}(p) - \mu_k^{2}(p)",
14: r"P_k(p) = \frac{\exp(-c_k(p))}{\sum_j \exp(-c_j(p))} , \qquad \hat{Z}(p) = \sum_k P_k(p)\, d_k",
15: r"C_d(p) = 1 - \frac{-\sum_k P_k(p) \log P_k(p)}{\log N_d}",
16: r"X_j(d) = d \cdot \left( K_j^{t} \right)^{-1} \bar{p}_j , \quad X_w(d) = \left( T_j^{t} \right)^{-1} X_j(d) , \quad X_i(d) = T_{i \leftarrow w}\, X_w(d)",
17: r"d'_{\min} = \max\left( 0.5 \cdot \min Z_L ,\; d_{\min} \right) , \qquad d'_{\max} = \min\left( 1.5 \cdot \max Z_L ,\; d_{\max} \right)",
# Masking after the softmax leaves the weights summing below one, so a pixel
 # darkens wherever a view is invalid. The implementation renormalizes -- and in
 # the plane-sweep path re-masks and renormalizes again after spatial smoothing
 # -- so the earlier form understated the method rather than describing it.
 18: r"w_i(p) = \frac{ M_i(p)\, e^{\ell_i(p)} }{ \sum_r M_r(p)\, e^{\ell_r(p)} } , \qquad I_c(p) = \sum_i w_i(p)\, \tilde{I}_i(p)",
19: r"\ell_i = -\frac{e_i}{0.05} + \log f_i + 2 \log s_i - \frac{b_i}{0.50} + 16 \cdot \mathbf{1}\left( s_i > 0.9999 \;\wedge\; b_i < 0.50 \right)",
# O collided with the output tuple O_j of Equation 1 and C was undefined at
 # point of use; G is geometry presence and kappa its confidence, both named in
 # the sentence that introduces this equation.
 20: r"M_{\mathrm{geo}} = \mathrm{clip}\left( (1-G(p)) + G(p)\,(1-\kappa(p)),\; 0,\; 1 \right) , \quad M = M_{\mathrm{geo}} \cdot \mathbf{1}\left[ Q(p) \neq \mathrm{observed} \right] , \quad \alpha_{\mathrm{eff}} = \alpha \cdot M",
21: r"\hat{I} = (1 - \alpha_{\mathrm{eff}})\, I_c + \alpha_{\mathrm{eff}} \cdot \mathrm{clip}\left( I_c + R,\; 0,\; 1 \right)",
22: r"\mathrm{ObservedCoverage} = \frac{|A|}{|\Omega|} , \qquad \mathrm{GateFill} = \frac{|H_{\mathrm{gate}}|}{|H|}",
23: r"\mathrm{GeneratedFraction} = \frac{|H_{\mathrm{generated}}|}{|\Omega|} , \qquad \mathrm{LockedChange} = \left| \{\, p \in A : \hat{I}(p) \neq I_{\mathrm{locked}}(p) \,\} \right|",
}

# Inline symbols. The key is the token written as $key$ in the manuscript source;
# the value is its LaTeX. Every one becomes a native Word inline equation.
INLINE = {
    # ---- cameras and calibration -------------------------------------------
    "I_i":        r"I_i",          "K_i": r"K_i",        "D_i": r"D_i",
    "T_i":        r"T_i",          "Kp_i": r"K'_i",      "P_set": r"P = \{X_n\}",
    "Kt_j":       r"K_j^{t}",      "Dt_j": r"D_j^{t}",   "Tt_j": r"T_j^{t}",
    "K_t":        r"K_t",          "Kp_t": r"K'_t",      "KL4_j": r"K_j^{\mathrm{L4}}",
    "C_w":        r"C_w = -R^{\top} t",
    "Xc":         r"X_c = T_{c \leftarrow w} X_w",
    "T_cw":       r"T_{c \leftarrow w} = [R \mid t]",
    # ---- rig ----------------------------------------------------------------
    "dC":         r"\Delta C",
    "Cbase":      r"C'_j = C_j^{\mathrm{base}} + \Delta C",
    "TLR":        r"T_{L \leftarrow R}^{\mathrm{view}}",
    # ---- depth and structure ------------------------------------------------
    "dDA3":       r"d_{\mathrm{DA3}}",   "dstar": r"d^*",
    "rtheta":     r"r_\theta",           "gp": r"g(p)",
    "g01":        r"g(p) \in [0,1]",
    "D32":        r"D = 32",
    "Zhat":       r"\hat{Z}",            "Cd": r"C_d",
    "ZL":         r"Z_L",
    # ---- fusion -------------------------------------------------------------
    "F_i":        r"F_i",          "Fik": r"F_{i,k}(p) = F_i(p_i(d_k))",
    "wi":         r"w_i",          "li": r"\ell_i",
    "ei":         r"e_i",          "fi": r"f_i",   "si": r"s_i",   "bi": r"b_i",
    "Ic":         r"I_c",          "Itil_i": r"\tilde{I}_i",  "Mi": r"M_i",
    # ---- composition --------------------------------------------------------
    "Rres":       r"R",            "alpha": r"\alpha",   "slog": r"s",
    "Opac":       r"O",            "Cconf": r"C",        "Mmask": r"M",
    "alphaeff":   r"\alpha_{\mathrm{eff}}",
    "Mgeo":       r"M_{\mathrm{geo}}",
    "Qp":         r"Q(p)",
    "Ilocked":    r"I_{\mathrm{locked}}",
    # ---- metric supports ----------------------------------------------------
    "Omega":      r"\Omega",       "Aset": r"A",   "Hset": r"H",
    "pj":         r"p_j",          "pt": r"p_t",
    "fx":         r"f_x",
    "fov":        r"2\arctan\left( w / 2 f_x \right)",
}
