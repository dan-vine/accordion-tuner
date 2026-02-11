"""
Musical temperaments - 32 built-in temperament definitions

Each temperament is defined as frequency ratios relative to the tonic for each
note in the chromatic scale (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B).

Ported from C++ built_in_temperaments.h
Copyright (C) 2019 Bill Farmer (original C++ implementation)
"""

from enum import IntEnum


class Temperament(IntEnum):
    """Available temperament presets (matches C++ ordering)"""
    KIRNBERGER_II = 0
    KIRNBERGER_III = 1
    WERCKMEISTER_III = 2
    WERCKMEISTER_IV = 3
    WERCKMEISTER_V = 4
    WERCKMEISTER_VI = 5
    BACH_KLAIS = 6
    JUST_BARBOUR = 7
    EQUAL = 8
    PYTHAGOREAN = 9
    VAN_ZWOLLE = 10
    MEANTONE_1_4 = 11
    SILBERMANN_1_6 = 12
    SALINAS_1_3 = 13
    ZARLINO_2_7 = 14
    ROSSI_1_5 = 15
    ROSSI_2_9 = 16
    RAMEAU_1_4 = 17
    KELLNER = 18
    VALLOTTI = 19
    YOUNG_II = 20
    BENDELER_III = 21
    NEIDHARDT_I = 22
    NEIDHARDT_II = 23
    NEIDHARDT_III = 24
    BRUDER_1829 = 25
    BARNES_1977 = 26
    LAMBERT_1774 = 27
    SCHLICK_VOGEL = 28
    MEANTONE_SHARP_1_4 = 29
    MEANTONE_FLAT_1_4 = 30
    LEHMAN_BACH = 31


# Display names for UI
TEMPERAMENT_NAMES = {
    Temperament.KIRNBERGER_II: "Kirnberger II",
    Temperament.KIRNBERGER_III: "Kirnberger III",
    Temperament.WERCKMEISTER_III: "Werckmeister III",
    Temperament.WERCKMEISTER_IV: "Werckmeister IV",
    Temperament.WERCKMEISTER_V: "Werckmeister V",
    Temperament.WERCKMEISTER_VI: "Werckmeister VI",
    Temperament.BACH_KLAIS: "Bach (Klais)",
    Temperament.JUST_BARBOUR: "Just (Barbour)",
    Temperament.EQUAL: "Equal Temperament",
    Temperament.PYTHAGOREAN: "Pythagorean",
    Temperament.VAN_ZWOLLE: "Van Zwolle",
    Temperament.MEANTONE_1_4: "Meantone (-1/4)",
    Temperament.SILBERMANN_1_6: "Silbermann (-1/6)",
    Temperament.SALINAS_1_3: "Salinas (-1/3)",
    Temperament.ZARLINO_2_7: "Zarlino (-2/7)",
    Temperament.ROSSI_1_5: "Rossi (-1/5)",
    Temperament.ROSSI_2_9: "Rossi (-2/9)",
    Temperament.RAMEAU_1_4: "Rameau (-1/4)",
    Temperament.KELLNER: "Kellner",
    Temperament.VALLOTTI: "Vallotti",
    Temperament.YOUNG_II: "Young II",
    Temperament.BENDELER_III: "Bendeler III",
    Temperament.NEIDHARDT_I: "Neidhardt I",
    Temperament.NEIDHARDT_II: "Neidhardt II",
    Temperament.NEIDHARDT_III: "Neidhardt III",
    Temperament.BRUDER_1829: "Bruder 1829",
    Temperament.BARNES_1977: "Barnes 1977",
    Temperament.LAMBERT_1774: "Lambert 1774",
    Temperament.SCHLICK_VOGEL: "Schlick (H. Vogel)",
    Temperament.MEANTONE_SHARP_1_4: "Meantone # (-1/4)",
    Temperament.MEANTONE_FLAT_1_4: "Meantone b (-1/4)",
    Temperament.LEHMAN_BACH: "Lehman-Bach",
}


# Temperament ratios for all 32 built-in temperaments
# Each row: C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B
# Ported from C++ built_in_temperaments.h
TEMPERAMENT_RATIOS = {
    # 0: Kirnberger II
    Temperament.KIRNBERGER_II: [
        1.000000000, 1.053497163, 1.125000000, 1.185185185,
        1.250000000, 1.333333333, 1.406250000, 1.500000000,
        1.580245745, 1.677050983, 1.777777778, 1.875000000
    ],

    # 1: Kirnberger III
    Temperament.KIRNBERGER_III: [
        1.000000000, 1.053497163, 1.118033989, 1.185185185,
        1.250000000, 1.333333333, 1.406250000, 1.495348781,
        1.580245745, 1.671850762, 1.777777778, 1.875000000
    ],

    # 2: Werckmeister III
    Temperament.WERCKMEISTER_III: [
        1.000000000, 1.053497942, 1.117403309, 1.185185185,
        1.252827249, 1.333333333, 1.404663923, 1.494926960,
        1.580246914, 1.670436332, 1.777777778, 1.879240873
    ],

    # 3: Werckmeister IV
    Temperament.WERCKMEISTER_IV: [
        1.000000000, 1.048750012, 1.119929822, 1.185185185,
        1.254242806, 1.333333333, 1.404663923, 1.493239763,
        1.573125018, 1.672323742, 1.785826183, 1.872885231
    ],

    # 4: Werckmeister V
    Temperament.WERCKMEISTER_V: [
        1.000000000, 1.057072991, 1.125000000, 1.189207115,
        1.257078722, 1.337858004, 1.414213562, 1.500000000,
        1.580246914, 1.681792831, 1.783810673, 1.885618083
    ],

    # 5: Werckmeister VI
    Temperament.WERCKMEISTER_VI: [
        1.000000000, 1.053497942, 1.114163307, 1.187481762,
        1.255862545, 1.333333333, 1.410112936, 1.497099016,
        1.580246914, 1.674483394, 1.781222643, 1.883793818
    ],

    # 6: Bach (Klais) - normalized Hz values
    Temperament.BACH_KLAIS: [
        262.76 / 440.0 * 1.681792831, 276.87 / 440.0 * 1.681792831,
        294.30 / 440.0 * 1.681792831, 311.46 / 440.0 * 1.681792831,
        328.70 / 440.0 * 1.681792831, 350.37 / 440.0 * 1.681792831,
        369.18 / 440.0 * 1.681792831, 393.70 / 440.0 * 1.681792831,
        415.30 / 440.0 * 1.681792831, 1.0,
        467.18 / 440.0 * 1.681792831, 492.26 / 440.0 * 1.681792831
    ],

    # 7: Just (Barbour) - normalized Hz values
    Temperament.JUST_BARBOUR: [
        264.00 / 440.0 * 1.681792831, 275.00 / 440.0 * 1.681792831,
        297.00 / 440.0 * 1.681792831, 316.80 / 440.0 * 1.681792831,
        330.00 / 440.0 * 1.681792831, 352.00 / 440.0 * 1.681792831,
        371.25 / 440.0 * 1.681792831, 396.00 / 440.0 * 1.681792831,
        412.50 / 440.0 * 1.681792831, 1.0,
        475.20 / 440.0 * 1.681792831, 495.00 / 440.0 * 1.681792831
    ],

    # 8: Equal Temperament
    Temperament.EQUAL: [
        1.000000000, 1.059463094, 1.122462048, 1.189207115,
        1.259921050, 1.334839854, 1.414213562, 1.498307077,
        1.587401052, 1.681792831, 1.781797436, 1.887748625
    ],

    # 9: Pythagorean
    Temperament.PYTHAGOREAN: [
        1.000000000, 1.067871094, 1.125000000, 1.185185185,
        1.265625000, 1.333333333, 1.423828125, 1.500000000,
        1.601806641, 1.687500000, 1.777777778, 1.898437500
    ],

    # 10: Van Zwolle
    Temperament.VAN_ZWOLLE: [
        1.000000000, 1.053497942, 1.125000000, 1.185185185,
        1.265625000, 1.333333333, 1.404663923, 1.500000000,
        1.580246914, 1.687500000, 1.777777778, 1.898437500
    ],

    # 11: Meantone (-1/4)
    Temperament.MEANTONE_1_4: [
        1.000000000, 1.044906727, 1.118033989, 1.196279025,
        1.250000000, 1.337480610, 1.397542486, 1.495348781,
        1.562500000, 1.671850762, 1.788854382, 1.869185977
    ],

    # 12: Silbermann (-1/6)
    Temperament.SILBERMANN_1_6: [
        1.000000000, 1.052506113, 1.120351187, 1.192569588,
        1.255186781, 1.336096753, 1.406250000, 1.496897583,
        1.575493856, 1.677050983, 1.785154534, 1.878886059
    ],

    # 13: Salinas (-1/3)
    Temperament.SALINAS_1_3: [
        1.000000000, 1.037362210, 1.115721583, 1.200000000,
        1.244834652, 1.338865900, 1.388888889, 1.493801582,
        1.549613310, 1.666666667, 1.792561899, 1.859535972
    ],

    # 14: Zarlino (-2/7)
    Temperament.ZARLINO_2_7: [
        1.000000000, 1.041666667, 1.117042372, 1.197872314,
        1.247783660, 1.338074130, 1.393827219, 1.494685500,
        1.556964062, 1.669627036, 1.790442378, 1.865044144
    ],

    # 15: Rossi (-1/5)
    Temperament.ROSSI_1_5: [
        1.000000000, 1.049459749, 1.119423732, 1.194051981,
        1.253109491, 1.336650124, 1.402760503, 1.496277870,
        1.570283397, 1.674968957, 1.786633554, 1.875000000
    ],

    # 16: Rossi (-2/9)
    Temperament.ROSSI_2_9: [
        1.000000000, 1.047433739, 1.118805855, 1.195041266,
        1.251726541, 1.337019165, 1.400438983, 1.495864870,
        1.566819334, 1.673582375, 1.787620248, 1.872413760
    ],

    # 17: Rameau (-1/4)
    Temperament.RAMEAU_1_4: [
        1.000000000, 1.051417112, 1.118033989, 1.179066456,
        1.250000000, 1.337480610, 1.401889482, 1.495348781,
        1.577125668, 1.671850762, 1.775938357, 1.869185977
    ],

    # 18: Kellner
    Temperament.KELLNER: [
        1.000000000, 1.053497942, 1.118918532, 1.185185185,
        1.251978681, 1.333333333, 1.404663923, 1.495940194,
        1.580246914, 1.673835206, 1.777777778, 1.877968022
    ],

    # 19: Vallotti
    Temperament.VALLOTTI: [
        1.000000000, 1.055879962, 1.119929822, 1.187864958,
        1.254242806, 1.336348077, 1.407839950, 1.496616064,
        1.583819943, 1.676104963, 1.781797436, 1.877119933
    ],

    # 20: Young II
    Temperament.YOUNG_II: [
        1.000000000, 1.053497942, 1.119929822, 1.185185185,
        1.254242806, 1.333333333, 1.404663923, 1.496616064,
        1.580246914, 1.676104963, 1.777777778, 1.877119933
    ],

    # 21: Bendeler III
    Temperament.BENDELER_III: [
        1.000000000, 1.057072991, 1.117403309, 1.185185185,
        1.257078722, 1.333333333, 1.409430655, 1.494926960,
        1.585609487, 1.676104963, 1.777777778, 1.879240873
    ],

    # 22: Neidhardt I
    Temperament.NEIDHARDT_I: [
        1.000000000, 1.055879962, 1.119929822, 1.186524315,
        1.254242806, 1.333333333, 1.407839950, 1.496616064,
        1.583819943, 1.676104963, 1.777777778, 1.879240873
    ],

    # 23: Neidhardt II
    Temperament.NEIDHARDT_II: [
        1.000000000, 1.057072991, 1.119929822, 1.187864958,
        1.255659964, 1.334839854, 1.411023157, 1.496616064,
        1.583819943, 1.676104963, 1.781797436, 1.883489946
    ],

    # 24: Neidhardt III
    Temperament.NEIDHARDT_III: [
        1.000000000, 1.057072991, 1.119929822, 1.187864958,
        1.255659964, 1.333333333, 1.411023157, 1.496616064,
        1.583819943, 1.676104963, 1.779786472, 1.883489946
    ],

    # 25: Bruder 1829
    Temperament.BRUDER_1829: [
        1.000000000, 1.056476308, 1.124364975, 1.187194447,
        1.253534828, 1.334086381, 1.409032810, 1.499576590,
        1.583819943, 1.678946488, 1.779786472, 1.879240873
    ],

    # 26: Barnes 1977
    Temperament.BARNES_1977: [
        1.000000000, 1.055879962, 1.119929822, 1.187864958,
        1.254242806, 1.336348077, 1.407839950, 1.496616064,
        1.583819943, 1.676104963, 1.781797436, 1.881364210
    ],

    # 27: Lambert 1774
    Temperament.LAMBERT_1774: [
        1.000000000, 1.055539344, 1.120652732, 1.187481762,
        1.255862545, 1.335916983, 1.407385792, 1.497099016,
        1.583309016, 1.677728102, 1.781222643, 1.880150581
    ],

    # 28: Schlick (H. Vogel)
    Temperament.SCHLICK_VOGEL: [
        1.000000000, 1.050646611, 1.118918532, 1.185185185,
        1.251978681, 1.336951843, 1.400862148, 1.495940194,
        1.575969916, 1.673835206, 1.782602458, 1.872885231
    ],

    # 29: Meantone # (-1/4)
    Temperament.MEANTONE_SHARP_1_4: [
        1.000000000, 1.044906727, 1.118033989, 1.168241235,
        1.250000000, 1.337480610, 1.397542486, 1.495348781,
        1.562500000, 1.671850762, 1.746928107, 1.869185977
    ],

    # 30: Meantone b (-1/4)
    Temperament.MEANTONE_FLAT_1_4: [
        1.000000000, 1.069984488, 1.118033989, 1.196279025,
        1.250000000, 1.337480610, 1.431083506, 1.495348781,
        1.600000000, 1.671850762, 1.788854382, 1.869185977
    ],

    # 31: Lehman-Bach
    Temperament.LEHMAN_BACH: [
        1.000000000, 1.058267368, 1.119929822, 1.187864958,
        1.254242806, 1.336348077, 1.411023157, 1.496616064,
        1.585609487, 1.676104963, 1.779786472, 1.881364210
    ],
}


def _ratios_to_cents(ratios: list[float]) -> list[float]:
    """Convert frequency ratios to cents offset from equal temperament"""
    import math
    equal = TEMPERAMENT_RATIOS[Temperament.EQUAL]
    return [1200 * math.log2(r / e) for r, e in zip(ratios, equal)]


# Pre-computed cents offsets from equal temperament (for display/comparison)
TEMPERAMENTS = {t: _ratios_to_cents(TEMPERAMENT_RATIOS[t]) for t in Temperament}
