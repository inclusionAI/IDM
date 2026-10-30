window.IDM_MATH = {
  "macros": {
    "\\ss": "\\mathbf{s}",
    "\\xx": "\\mathbf{x}",
    "\\cc": "\\mathbf{c}",
    "\\zz": "\\mathbf{z}",
    "\\vv": "\\mathbf{v}",
    "\\dd": "\\mathbf{d}",
    "\\hh": "\\mathbf{h}",
    "\\gv": "\\mathbf{g}",
    "\\mmF": "\\boldsymbol{F}",
    "\\mepsilon": "\\boldsymbol{\\epsilon}",
    "\\cL": "\\mathcal{L}",
    "\\cN": "\\mathcal{N}",
    "\\cQ": "\\mathcal{Q}",
    "\\method": "IDM",
    "\\sg": "\\mathop{\\operatorname{sg}}\\nolimits",
    "\\diag": "\\operatorname{diag}"
  },
  "equations": {
    "eq:score_velocity": {
      "tex": "\\ss_t(\\xx_t)=-\\frac{\\xx_t+(1-t)\\vv_t^*(\\xx_t)}{t}\\,.",
      "number": "1"
    },
    "eq:reverse_kl_field": {
      "tex": "\\gv_{\\mathrm{KL}}(\\xx\\mid\\cc)=\\ss_\\theta(\\xx\\mid\\cc)-\\ss_r(\\xx\\mid\\cc).",
      "number": "3"
    },
    "eq:score_decomp_main": {
      "tex": "\\ss_p(\\xx)\n  =\\underbrace{\\ss_{q_p}(\\xx)}_{=-\\Sigma_p^{-1}(\\xx-\\mu_p)}\n  +\\underbrace{\\boldsymbol{\\eta}_p(\\xx)}_{\\text{non-Gaussian score residual}}\\,,",
      "number": "6"
    },
    "eq:grad_decomp_main": {
      "tex": "\\gv_{\\mathrm{KL}}(\\xx\\mid\\cc)\n  =\n  \\underbrace{\\left(\\ss_{q_\\theta}(\\xx\\mid\\cc)-\\ss_{q_r}(\\xx\\mid\\cc)\\right)}_{\\gv_{\\mathrm{KL}}^{\\mathrm{G}}:\\;\\text{Gaussian moment component}}\n  +\n  \\underbrace{\\left(\\boldsymbol{\\eta}_\\theta(\\xx\\mid\\cc)-\\boldsymbol{\\eta}_r(\\xx\\mid\\cc)\\right)}_{\\gv_{\\mathrm{KL}}^{\\mathrm{nG}}:\\;\\text{non-Gaussian residual component}}.",
      "number": "7"
    },
    "eq:d_micro": {
      "tex": "\\dd_t^*(\\xx_t)\n  := \\vv^*_{\\theta,t}(\\xx_t) - \\vv^*_{r,t}(\\xx_t)\n  = \\frac{t}{1{-}t}\\left(\\ss_{r,t}(\\xx_t) - \\ss_{\\theta,t}(\\xx_t)\\right)\\!,",
      "number": "9"
    },
    "eq:branch_instantiation": {
      "tex": "\\widehat\\vv_{r,t}(\\xx_t\\mid\\cc):=\\mmF_\\theta(\\xx_t,t,\\cc),\\qquad\n  \\widehat\\vv_{\\theta,t}(\\xx_t\\mid\\cc):=\\mmF_\\theta(\\xx_t,t,-\\cc),",
      "number": "10"
    },
    "eq:dmd_targets": {
      "tex": "\\begin{aligned}\n\\vv_t^{\\mathrm{DMD}}\n&:=\\sg\\!\\left(\\mmF_\\theta(\\xx_t,t,\\cc)-\\widehat\\dd_{t'}^{\\mathrm{DMD}}\\right),\\\\\n\\xx_0^{\\mathrm{DMD}}\n&:=\\sg\\!\\left(\\xx_t-t\\vv_t^{\\mathrm{DMD}}\\right).\n\\end{aligned}",
      "number": "11"
    },
    "eq:l_score": {
      "tex": "\\cL_{\\text{score}}(\\theta)\n:=\\mathbb E\\!\\left[\n\\underbrace{\\|\\mmF_\\theta(\\xx_t,t,\\cc)-\\vv_t^{\\mathrm{DMD}}\\|_2^2}_{\\ell_v(t)}\n+\\underbrace{\\|\\widehat\\xx_0(\\xx_t,t,\\cc)-\\xx_0^{\\mathrm{DMD}}\\|_2^2}_{\\ell_x(t)}\n\\right].",
      "number": "12"
    },
    "eq:moment_loss": {
      "tex": "\\cL_{\\text{pop}} = \\|\\mu_\\theta^\\phi - \\mu_r^\\phi\\|_2^2 + \\sum_{j=1}^{d_\\phi} \\left(\\sqrt{\\nu_{\\theta,j}^\\phi} - \\sqrt{\\nu_{r,j}^\\phi}\\right)^2\\!.",
      "number": "13"
    },
    "eq:l_idm": {
      "tex": "\\cL_{\\text{\\method}}(\\theta)\n=\\cL_{\\text{base}}(\\theta)\n+\\lambda_{\\text{score}}\\cL_{\\text{score}}(\\theta)\n+\\lambda_{\\text{pop}}\\widehat{\\cL}_{\\text{pop}} .",
      "number": "14"
    }
  }
};
window.IDM_CITATION = "@misc{liu2026intrinsicdistributionmatching,\n  title={Intrinsic Distribution Matching},\n  author={Deyuan Liu and Yansen Han and Kaiwen Zheng and Jun Xie and Jiacheng Liu and Zhenglin Cheng and Peng Sun and Jintao Zhang and Jianguo Li and Tao Lin},\n  year={2026},\n  eprint={2610.39633},\n  archivePrefix={arXiv},\n  primaryClass={cs.CV},\n  url={https://arxiv.org/abs/2610.39633},\n}\n\n@article{liu2026self,\n  title={Self-Adversarial One Step Generation via Condition Shifting},\n  author={Liu, Deyuan and Sun, Peng and Han, Yansen and Cheng, Zhenglin and Chen, Chuyan and Lin, Tao},\n  journal={arXiv preprint arXiv:2604.12322},\n  year={2026}\n}";
