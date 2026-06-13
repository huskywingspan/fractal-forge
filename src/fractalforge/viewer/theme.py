"""Premium Dear PyGui theme using the Infinite Descent brand palette.

Brand: deep navy background (#0a0e1a), cyan accent (#00d4ff), violet (#a855f7).
A single global theme plus a couple of accent themes for badges/buttons.
"""

import dearpygui.dearpygui as dpg

# Brand palette (RGBA, 0-255)
BG = (10, 14, 26, 255)
BG_PANEL = (18, 24, 40, 255)
BG_INPUT = (26, 33, 52, 255)
BG_HOVER = (36, 46, 72, 255)
CYAN = (0, 212, 255, 255)
CYAN_DIM = (0, 150, 190, 255)
VIOLET = (168, 85, 247, 255)
VIOLET_DIM = (120, 60, 180, 255)
TEXT = (226, 232, 240, 255)
TEXT_DIM = (140, 152, 176, 255)
BORDER = (40, 52, 80, 255)


def apply_theme():
    """Build and bind the global viewer theme. Returns the theme tag."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Colors
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, BG_PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_Border, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Text, TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, TEXT_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BG_INPUT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, BG_HOVER)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, BG)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BG)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (32, 42, 66, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, CYAN_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, CYAN)
            dpg.add_theme_color(dpg.mvThemeCol_Header, VIOLET_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, VIOLET_DIM)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, VIOLET)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, CYAN)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, CYAN)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, CYAN)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, BG)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, BORDER)
            # Geometry: rounded, breathing room
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 5)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 3)

    dpg.bind_theme(global_theme)
    return global_theme


def accent_button_theme(color):
    """A filled accent button theme (e.g. for the primary action)."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                tuple(min(c + 30, 255) for c in color[:3]) + (255,))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color)
            dpg.add_theme_color(dpg.mvThemeCol_Text, BG)
    return t
