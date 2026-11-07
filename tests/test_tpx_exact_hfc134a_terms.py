"""HFC-134a Helmholtz sums follow the pinned grouped-term equations."""

import math

import pytest

from cantera_python.tpx_exact import HFC134a


def _source_residual_sums(phase):
    tau = phase.Tc / phase.T
    delta = phase.Rho / phase.Roc
    ranges = ((0, 8, 0), (8, 11, 1), (11, 17, 2),
              (17, 20, 3), (20, 21, 4))
    phi_r = 0.0
    phi_r_tau = 0.0
    phi_r_delta = 0.0
    for first, last, damping_power in ranges:
        damping = 1.0 if damping_power == 0 else math.exp(-delta**damping_power)
        phi_r += damping * sum(
            phase.a[i] * tau**phase.t[i] * delta**phase.d[i]
            for i in range(first, last)
        )
        phi_r_tau += damping * sum(
            phase.a[i] * phase.t[i] * tau**phase.t[i] * delta**phase.d[i]
            for i in range(first, last)
        )
        phi_r_delta += damping * sum(
            phase.a[i] * tau**phase.t[i] * (phase.d[i] - damping_power*delta**damping_power)
            * delta**(phase.d[i] - 1)
            for i in range(first, last)
        )
    return tau, delta, phi_r, phi_r_tau, phi_r_delta


@pytest.mark.parametrize("temperature,density", [(180.0, 0.08), (200.0, 638.0),
                                                   (300.0, 31.0), (360.0, 500.0)])
def test_hfc134a_free_energy_and_energy_use_undamped_first_group(temperature, density):
    phase = HFC134a()
    phase.T = temperature
    phase.Rho = density
    tau, delta, phi_r, phi_r_tau, phi_r_delta = _source_residual_sums(phase)
    b = phase.b
    phi_0 = (b[0] + b[1]*tau + b[2]*math.log(tau) + math.log(delta)
             + b[3]/math.sqrt(tau) + b[4]*tau**-0.75)
    phi_0_tau = (b[1]*tau + b[2] - 0.5*b[3]*tau**-0.5
                 - 0.75*b[4]*tau**-0.75)

    expected_f = phase.R * temperature * (phi_0 + phi_r)
    expected_u = phase.R * temperature * (phi_0_tau + phi_r_tau)
    expected_p = (phase.R * temperature * delta**2 * phase.Roc
                  * (phi_r_delta + 1.0/delta))

    assert phase.fp() == pytest.approx(expected_f, rel=2e-14, abs=1e-10)
    assert phase.up() == pytest.approx(expected_u, rel=2e-14, abs=1e-10)
    assert phase.Pp() == pytest.approx(expected_p, rel=2e-14, abs=1e-10)
