from setuptools import setup

package_name = "mocap_natnet"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    python_requires=">=3.10",
    zip_safe=True,
    maintainer="jason",
    maintainer_email="jasonsiu@caltech.edu",
    description="Minimal OptiTrack NatNet 3.x/4.x rigid-body client (standard library only).",
    license="MIT",
    tests_require=["pytest"],
)
