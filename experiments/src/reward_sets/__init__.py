"""Fail-closed reward entry points for individual training recipes.

Each public module declares the exact ``metadata.verifier`` values that its
dataset is allowed to contain. Training recipes should point at one of these
entry points instead of the broad all-domain diagnostic dispatcher.
"""
