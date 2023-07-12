import cartoon_scenarios, seasonal_anim, secular_anim, het_vel_anim, comparison

params = {
    'P': 90,
    'interval': 1,
    'wavelength': 0.056,
    'suffix': '_cBand'
}

sample_bias_at = 15

het_vel_params = {
    'trough_p': 0.2,
    'trough_vel': -50,
    'center_p': 0.8,
    'center_vel': -25,
}

seasonal_params = {
    'seasonal_reg': True,
}

cartoon_scenarios.run(**params, **het_vel_params)
seasonal_anim.run(**params, **seasonal_params)
secular_anim.run(**params, sample_bias_at=sample_bias_at)
het_vel_anim.run(**params, **het_vel_params, sample_bias_at=sample_bias_at)
comparison.run(**params)