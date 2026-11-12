/*
 * test.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Sep 15 2026, 20:56 by COMSOL 5.4.0.388. */
public class test {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("/home/germanf");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "geom1");

    model.study().create("std1");
    model.study("std1").setGenConv(true);
    model.study("std1").create("freq", "Frequency");
    model.study("std1").feature("freq").set("solnum", "auto");
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("ngen", "5");
    model.study("std1").feature("freq").activate("ewfd", true);

    model.param().set("f0", "10[GHZ]");
    model.param().remove("f0");
    model.param().set("f0", "10[GHz]");
    model.param().descr("f0", "operating frequency");
    model.param().set("c0", "c_const");
    model.param().descr("c0", "speed of light");
    model.param().set("lambda0", "c0/f0");
    model.param().descr("lambda0", "free-space wavelength");
    model.param().set("k0", "2*pi/lambda0");
    model.param().descr("k0", "free-space wavenumber");
    model.param().set("E0", "1[V/m]");
    model.param().descr("E0", "plane-wave amplitude");

    model.study("std1").feature("freq").set("plist", "f0");
    model.study("std1").feature("freq").set("punit", "Hz");

    model.component("comp1").geom("geom1").run();
    model.component("comp1").geom("geom1").create("sph1", "Sphere");
    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0/4");
    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").create("blk1", "Block");
    model.component("comp1").geom("geom1").feature("blk1").set("pos", new String[]{"", "0", "0"});
    model.component("comp1").geom("geom1").feature("blk1").set("size", new String[]{"lambda0", "lambda0", "1"});
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 2);
    model.component("comp1").geom("geom1").feature("blk1").set("base", "center");
    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").feature("blk1").set("pos", new int[]{0, 0, 0});

    model.label("test.mph");

    model.component("comp1").geom("geom1").run("blk1");

    model.label("test.mph");

    model.component("comp1").geom("geom1").feature("fin").set("action", "assembly");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("eta", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("Cp", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("rho", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("k", "Piecewise");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("cs", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("an1", "Analytic");
    model.component("comp1").material("mat1").propertyGroup("def").func().create("an2", "Analytic");
    model.component("comp1").material("mat1").propertyGroup().create("RefractiveIndex", "Refractive index");
    model.component("comp1").material("mat1").propertyGroup().create("NonlinearModel", "Nonlinear model");
    model.component("comp1").material("mat1").label("Air");
    model.component("comp1").material("mat1").set("family", "air");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-8.38278E-7+8.35717342E-8*T^1-7.69429583E-11*T^2+4.6437266E-14*T^3-1.06585607E-17*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("eta").set("fununit", "Pa*s");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp")
         .set("pieces", new String[][]{{"200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("Cp").set("fununit", "J/(kg*K)");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("expr", "pA*0.02897/R_const[K*mol/J]/T");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("dermethod", "manual");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("argders", new String[][]{{"pA", "d(pA*0.02897/R_const/T,pA)"}, {"T", "d(pA*0.02897/R_const/T,T)"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("argunit", "Pa,K");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("fununit", "kg/m^3");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("plotargs", new String[][]{{"pA", "0", "1"}, {"T", "0", "1"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("k")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("fununit", "W/(m*K)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("expr", "sqrt(1.4*R_const[K*mol/J]/0.02897*T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("dermethod", "manual");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("fununit", "m/s");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("plotargs", new String[][]{{"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("funcname", "alpha_p");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("expr", "-1/rho(pA,T)*d(rho(pA,T),T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("argunit", "Pa,K");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("fununit", "1/K");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("plotargs", new String[][]{{"pA", "101325", "101325"}, {"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("funcname", "muB");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("expr", "0.6*eta(T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("fununit", "Pa*s");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2")
         .set("plotargs", new String[][]{{"T", "200", "1600"}});
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("relpermittivity", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("def").set("dynamicviscosity", "eta(T)");
    model.component("comp1").material("mat1").propertyGroup("def").set("ratioofspecificheat", "1.4");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("electricconductivity", new String[]{"0[S/m]", "0", "0", "0", "0[S/m]", "0", "0", "0", "0[S/m]"});
    model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", "Cp(T)");
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "rho(pA,T)");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalconductivity", new String[]{"k(T)", "0", "0", "0", "k(T)", "0", "0", "0", "k(T)"});
    model.component("comp1").material("mat1").propertyGroup("def").set("soundspeed", "cs(T)");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)"});
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "0.02897");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "muB(T)");
    model.component("comp1").material("mat1").propertyGroup("def").addInput("temperature");
    model.component("comp1").material("mat1").propertyGroup("def").addInput("pressure");
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("n", "");
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex").set("ki", "");
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex")
         .set("n", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex")
         .set("ki", new String[]{"0", "0", "0", "0", "0", "0", "0", "0", "0"});
    model.component("comp1").material("mat1").propertyGroup("NonlinearModel").set("BA", "(def.gamma+1)/2");
    model.component("comp1").material("mat1").materialType("nonSolid");
    model.component("comp1").material("mat1").set("family", "air");

    model.component("comp1").physics("ewfd").create("sctr1", "Scattering", 2);
    model.component("comp1").physics("ewfd").feature("sctr1").selection().set(1, 2, 3, 4, 5, 6);
    model.component("comp1").physics("ewfd").feature("sctr1").set("IncidentField", "EField");
    model.component("comp1").physics("ewfd").feature("sctr1").set("E0i", new String[]{"E0", "0", "0"});
    model.component("comp1").physics("ewfd").feature("sctr1").set("kdir", new String[]{"0", "0", "-k0"});

    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").physics("ewfd").create("cont1", "Continuity", 2);
    model.component("comp1").physics("ewfd").feature().remove("cont1");

    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").physics("ewfd").create("trans1", "TransitionBoundaryCondition", 2);

    model.component("comp1").view("view1").set("hidestatus", "hide");
    model.component("comp1").view("view1").set("scenelight", true);
    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").physics("ewfd").feature().remove("trans1");

    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0/2", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0/2", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0/2", 2);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 2);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("dif1", "Difference");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").init();
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").init();
    model.component("comp1").geom("geom1").feature("dif1").selection("input2").set();
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature().move("blk1", 0);
    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").feature().remove("dif1");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").feature().move("blk1", 1);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("dif1", "Difference");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature("sph1").set("selresult", true);
    model.component("comp1").geom("geom1").feature("sph1").set("color", "2");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").feature("dif1").selection("input2").named("sph1");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run("dif1");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature().remove("dif1");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").feature("sph1").set("selresult", false);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("dif1", "Difference");
    model.component("comp1").geom("geom1").feature("sph1").set("selresult", true);
    model.component("comp1").geom("geom1").feature().remove("dif1");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").feature("sph1").set("selresultshow", "bnd");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").runPre("fin");

    model.component("comp1").view("view1").set("transparency", false);
    model.component("comp1").view("view1").set("renderwireframe", true);
    model.component("comp1").view("view1").set("transparency", true);
    model.component("comp1").view("view1").set("scenelight", false);
    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").view("view1").set("transparency", false);
    model.component("comp1").view("view1").set("renderwireframe", false);
    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").physics("ewfd").create("trans1", "TransitionBoundaryCondition", 2);
    model.component("comp1").physics("ewfd").feature("trans1").selection().named("geom1_sph1_bnd");
    model.component("comp1").physics("ewfd").feature("trans1").selection().set(7, 8, 9, 10, 11, 12, 13, 14);
    model.component("comp1").physics("ewfd").feature().remove("trans1");
    model.component("comp1").physics("ewfd").create("trans1", "TransitionBoundaryCondition", 2);
    model.component("comp1").physics("ewfd").feature("trans1").selection().named("geom1_sph1_bnd");
    model.component("comp1").physics("ewfd").feature("trans1").selection().set();
    model.component("comp1").physics("ewfd").feature("trans1").selection().all();
    model.component("comp1").physics("ewfd").feature("trans1").selection()
         .set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    model.component("comp1").physics("ewfd").feature("trans1").selection().all();
    model.component("comp1").physics("ewfd").feature("trans1").selection().set();
    model.component("comp1").physics("ewfd").feature("trans1").selection().named("geom1_sph1_bnd");

    model.component("comp1").geom("geom1").feature("sph1").set("selresultshow", "all");
    model.component("comp1").geom("geom1").run("fin");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").physics("ewfd").feature("trans1").selection().named("geom1_sph1_bnd");
    model.component("comp1").physics("ewfd").feature("trans1").selection().set();
    model.component("comp1").physics("ewfd").feature("trans1").selection().named("geom1_sph1_bnd");
    model.component("comp1").physics("ewfd").feature("trans1").selection().set();

    model.component("comp1").geom("geom1").feature("sph1").set("selresultshow", "dom");
    model.component("comp1").geom("geom1").feature("sph1").set("color", "2");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("dif1", "Difference");
    model.component("comp1").geom("geom1").feature("dif1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature("dif1").selection("input2").named("sph1");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature().remove("dif1");
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("uni1", "Union");
    model.component("comp1").geom("geom1").feature("uni1").selection("input").set("blk1");
    model.component("comp1").geom("geom1").feature("uni1").selection("input").named("sph1");
    model.component("comp1").geom("geom1").feature("uni1").selection("input").named("");
    model.component("comp1").geom("geom1").feature("uni1").selection("input").set("blk1", "sph1");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run();

    model.component("comp1").material("mat1").selection().named("geom1_sph1_dom");
    model.component("comp1").material("mat1").selection().all();

    model.component("comp1").physics("ewfd").feature("trans1").selection().set(6, 7, 11);
    model.component("comp1").physics("ewfd").feature("trans1").selection().all();
    model.component("comp1").physics("ewfd").feature("trans1").selection().set(6, 8, 9, 12, 13);

    model.param().set("chi_ee", "0.1");
    model.param().descr("chi_ee", "Electric susceptibility");

    model.component("comp1").physics("ewfd").create("scu1", "SurfaceCurrent", 2);

    model.component("comp1").coordSystem().create("sys2", "Spherical");

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys2");

    model.component("comp1").view("view1").set("renderwireframe", false);
    model.component("comp1").view("view1").set("hidestatus", "hide");
    model.component("comp1").view("view1").hideEntities().create("hide1");
    model.component("comp1").view("view1").hideEntities("hide1").geom(2);
    model.component("comp1").view("view1").hideEntities("hide1").add(1);
    model.component("comp1").view("view1").hideEntities("hide1").add(4);
    model.component("comp1").view("view1").hideEntities("hide1").add(2);
    model.component("comp1").view("view1").set("hidestatus", "hide");

    model.component("comp1").physics("ewfd").feature("scu1").selection().set(7, 13);
    model.component("comp1").physics("ewfd").feature("scu1").selection().all();
    model.component("comp1").physics("ewfd").feature("scu1").selection().set(6, 7, 8, 9, 10, 11, 12, 13);
    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Etheta/", "0"});

    model.param().set("eta", "-100*j[ohm]");

    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Etheta/eta", "0"});

    model.result().numerical().create("pev1", "EvalPoint");

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "GlobalSystem");
    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Ex/eta", "0"});
    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys2");
    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Etheta/eta", "0"});
    model.component("comp1").physics("ewfd").feature().remove("trans1");
    model.component("comp1").physics("ewfd").create("trans1", "TransitionBoundaryCondition", 2);
    model.component("comp1").physics("ewfd").feature("trans1").set("DisplacementFieldModel", "RefractiveIndex");
    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys2");
    model.component("comp1").physics("ewfd").feature().remove("trans1");

    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "ewfd.Ethetaruyrtu/eta", "0"});

    model.component("comp1").mesh("mesh1").run();

    model.sol().create("sol1");
    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Etheta/eta", "0"});

    model.label("test.mph");

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Ex/eta", "0"});

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");

    return model;
  }

  public static Model run2(Model model) {
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").feature("size").set("custom", false);
    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 6);

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 9);
    model.component("comp1").mesh("mesh1").run();

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").physics("ewfd").feature("scu1").active(false);

    model.component("comp1").material("mat1").selection().set(1);
    model.component("comp1").material().create("mat2", "Common");
    model.component("comp1").material("mat2").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material("mat2").label("FR4 (Circuit Board)");
    model.component("comp1").material("mat2").set("family", "custom");
    model.component("comp1").material("mat2").set("specular", "custom");
    model.component("comp1").material("mat2")
         .set("customspecular", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("diffuse", "custom");
    model.component("comp1").material("mat2")
         .set("customdiffuse", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("ambient", "custom");
    model.component("comp1").material("mat2")
         .set("customambient", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("noise", true);
    model.component("comp1").material("mat2").set("noisescale", 0.05);
    model.component("comp1").material("mat2").set("noisefreq", 2.1);
    model.component("comp1").material("mat2").set("lighting", "cooktorrance");
    model.component("comp1").material("mat2").set("fresnel", 0.3);
    model.component("comp1").material("mat2").set("roughness", 0.6);
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("electricconductivity", new String[]{"0.004[S/m]", "0", "0", "0", "0.004[S/m]", "0", "0", "0", "0.004[S/m]"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermittivity", new String[]{"4.5", "0", "0", "0", "4.5", "0", "0", "0", "4.5"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"18e-6[1/K]", "0", "0", "0", "18e-6[1/K]", "0", "0", "0", "18e-6[1/K]"});
    model.component("comp1").material("mat2").propertyGroup("def").set("heatcapacity", "1369[J/(kg*K)]");
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "1900[kg/m^3]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("thermalconductivity", new String[]{"0.3[W/(m*K)]", "0", "0", "0", "0.3[W/(m*K)]", "0", "0", "0", "0.3[W/(m*K)]"});
    model.component("comp1").material("mat2").propertyGroup("Enu").set("youngsmodulus", "22e9[Pa]");
    model.component("comp1").material("mat2").propertyGroup("Enu").set("poissonsratio", "0.15");
    model.component("comp1").material("mat2").set("family", "custom");
    model.component("comp1").material("mat2").set("lighting", "cooktorrance");
    model.component("comp1").material("mat2").set("fresnel", 0.3);
    model.component("comp1").material("mat2").set("roughness", 0.6);
    model.component("comp1").material("mat2").set("ambient", "custom");
    model.component("comp1").material("mat2")
         .set("customambient", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("diffuse", "custom");
    model.component("comp1").material("mat2")
         .set("customdiffuse", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("specular", "custom");
    model.component("comp1").material("mat2")
         .set("customspecular", new double[]{0.39215686274509803, 0.7843137254901961, 0.39215686274509803});
    model.component("comp1").material("mat2").set("noisescale", 0.05);
    model.component("comp1").material("mat2").set("noise", "on");
    model.component("comp1").material("mat2").set("noisefreq", 2.1);
    model.component("comp1").material("mat2").set("noise", "on");
    model.component("comp1").material("mat2").set("alpha", 1);
    model.component("comp1").material("mat2").selection().all();
    model.component("comp1").material("mat2").selection().set(2);
    model.component("comp1").material().remove("mat2");
    model.component("comp1").material().create("mat2", "Common");
    model.component("comp1").material("mat2").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material("mat2").propertyGroup().create("RefractiveIndex", "Refractive index");
    model.component("comp1").material("mat2").label("Silicon");
    model.component("comp1").material("mat2").set("family", "custom");
    model.component("comp1").material("mat2").set("specular", "custom");
    model.component("comp1").material("mat2").set("customspecular", new double[]{0.7843137254901961, 1, 1});
    model.component("comp1").material("mat2").set("diffuse", "custom");
    model.component("comp1").material("mat2")
         .set("customdiffuse", new double[]{0.6666666666666666, 0.6666666666666666, 0.7058823529411765});
    model.component("comp1").material("mat2").set("ambient", "custom");
    model.component("comp1").material("mat2")
         .set("customambient", new double[]{0.6666666666666666, 0.6666666666666666, 0.7058823529411765});
    model.component("comp1").material("mat2").set("noise", true);
    model.component("comp1").material("mat2").set("noisefreq", 1);
    model.component("comp1").material("mat2").set("lighting", "cooktorrance");
    model.component("comp1").material("mat2").set("fresnel", 0.7);
    model.component("comp1").material("mat2").set("roughness", 0.5);
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermeability", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("electricconductivity", new String[]{"1e-12[S/m]", "0", "0", "0", "1e-12[S/m]", "0", "0", "0", "1e-12[S/m]"});
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"2.6e-6[1/K]", "0", "0", "0", "2.6e-6[1/K]", "0", "0", "0", "2.6e-6[1/K]"});
    model.component("comp1").material("mat2").propertyGroup("def").set("heatcapacity", "700[J/(kg*K)]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("relpermittivity", new String[]{"11.7", "0", "0", "0", "11.7", "0", "0", "0", "11.7"});
    model.component("comp1").material("mat2").propertyGroup("def").set("density", "2329[kg/m^3]");
    model.component("comp1").material("mat2").propertyGroup("def")
         .set("thermalconductivity", new String[]{"130[W/(m*K)]", "0", "0", "0", "130[W/(m*K)]", "0", "0", "0", "130[W/(m*K)]"});
    model.component("comp1").material("mat2").propertyGroup("Enu").set("youngsmodulus", "170e9[Pa]");
    model.component("comp1").material("mat2").propertyGroup("Enu").set("poissonsratio", "0.28");
    model.component("comp1").material("mat2").propertyGroup("RefractiveIndex").set("n", "");
    model.component("comp1").material("mat2").propertyGroup("RefractiveIndex").set("ki", "");
    model.component("comp1").material("mat2").propertyGroup("RefractiveIndex")
         .set("n", new String[]{"3.48", "0", "0", "0", "3.48", "0", "0", "0", "3.48"});
    model.component("comp1").material("mat2").propertyGroup("RefractiveIndex")
         .set("ki", new String[]{"0", "0", "0", "0", "0", "0", "0", "0", "0"});
    model.component("comp1").material("mat2").set("family", "custom");
    model.component("comp1").material("mat2").set("lighting", "cooktorrance");
    model.component("comp1").material("mat2").set("fresnel", 0.7);
    model.component("comp1").material("mat2").set("roughness", 0.5);
    model.component("comp1").material("mat2").set("ambient", "custom");
    model.component("comp1").material("mat2")
         .set("customambient", new double[]{0.6666666666666666, 0.6666666666666666, 0.7058823529411765});
    model.component("comp1").material("mat2").set("diffuse", "custom");
    model.component("comp1").material("mat2")
         .set("customdiffuse", new double[]{0.6666666666666666, 0.6666666666666666, 0.7058823529411765});
    model.component("comp1").material("mat2").set("specular", "custom");
    model.component("comp1").material("mat2").set("customspecular", new double[]{0.7843137254901961, 1, 1});
    model.component("comp1").material("mat2").set("noisescale", 0);
    model.component("comp1").material("mat2").set("noise", "on");
    model.component("comp1").material("mat2").set("noisefreq", 1);
    model.component("comp1").material("mat2").set("noise", "on");
    model.component("comp1").material("mat2").set("alpha", 1);
    model.component("comp1").material("mat2").selection().named("geom1_sph1_dom");

    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").view("view1").set("hidestatus", "hide");
    model.component("comp1").view("view1").hideEntities().create("hide2");
    model.component("comp1").view("view1").hideEntities("hide2").geom(3);
    model.component("comp1").view("view1").hideEntities("hide2").add(1);
    model.component("comp1").view("view1").hideObjects().clear();
    model.component("comp1").view("view1").hideEntities().clear();
    model.component("comp1").view("view1").hideMesh().clear();
    model.component("comp1").view("view1").set("hidestatus", "hide");
    model.component("comp1").view("view1").hideEntities().create("hide1");
    model.component("comp1").view("view1").hideEntities("hide1").geom(2);
    model.component("comp1").view("view1").hideEntities("hide1").add(2);
    model.component("comp1").view("view1").hideEntities("hide1").add(1);
    model.component("comp1").view("view1").hideEntities("hide1").add(4);
    model.component("comp1").view("view1").set("hidestatus", "hide");
    model.component("comp1").view("view1").set("scenelight", false);

    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").run();

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").automatic(true);

    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.study("std1").feature("freq").set("usestoresel", "all");
    model.study("std1").feature("freq").set("errestandadap", "none");

    model.sol("sol1").feature("v1").set("resscalemethod", "manual");
    model.sol("sol1").feature("v1").set("scalemethod", "manual");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "lambda0/4");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").feature("size").set("hmax", "lambda0/4");
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "lambda0/8");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result().remove("pg1");

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lambda0/4");
    model.component("comp1").mesh("mesh1").feature("size1").set("hmin", "lambda0/4");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (ewfd)");
    model.result("pg1").set("showlooplevel", new String[]{"off", "off", "off"});
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("data", "dset1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").set("colortable", "RainbowLight");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("size1").set("hmin", "lambda0/8");
    model.component("comp1").mesh("mesh1").feature("ftet1").set("method", "dellegacy52");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(true);

    model.sol("sol1").feature("v1").set("scalemethod", "none");
    model.sol("sol1").feature("v1").set("resscalemethod", "auto");
    model.sol("sol1").feature("v1").set("scalemethod", "auto");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "dDef");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    return model;
  }

  public static Model run3(Model model) {

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature("fc1").set("termonres", "auto");
    model.sol("sol1").feature("s1").feature("fc1").set("reserrfact", 100);
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("fc1").set("reserrfact", 1000);
    model.sol("sol1").feature("s1").feature("i1").set("irestol", 0.1);
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("i1").set("irestol", 0.01);
    model.sol("sol1").feature("s1").feature("fc1").set("minstep", "1.0E-1");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("fc1").set("minstep", 1.0E-4);
    model.sol("sol1").feature("s1").feature("fc1").set("ntermauto", "tol");
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("i1").set("irestol", 0.1);
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("i1").set("irestol", 0.01);
    model.sol("sol1").feature("s1").feature("fc1").set("ntolfact", 100);
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("fc1").set("ntolfact", 1);
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").set("errorchk", "auto");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "dDef");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys1");
    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "ewfd.Eetha/eta", "0"});
    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "GlobalSystem");

    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys1");
    model.component("comp1").physics("ewfd").feature("scu1").set("Js0", new String[]{"0", "(sys2.)/eta", "0"});
    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "GlobalSystem");
    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "(comp1.sys2.e_phi2 + )/eta", "0"});

    model.result("pg1").run();
    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "(sys2.e_phi2 + sys2.e_theta2)/eta", "0"});

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg1").run();

    model.variable().create("var1");
    model.variable().remove("var1");

    model.result("pg1").run();
    model.result("pg1").run();
    model.result().create("pg2", "PlotGroup2D");
    model.result("pg2").run();
    model.result("pg2").active(false);
    model.result("pg2").run();
    model.result().remove("pg2");
    model.result("pg1").run();

    model.component("comp1").coordSystem().create("sys3", "Mapping");
    model.component("comp1").coordSystem().remove("sys3");
    model.component("comp1").coordSystem("sys1").set("frametype", "material");
    model.component("comp1").coordSystem("sys1").set("reversenormal", false);
    model.component("comp1").coordSystem("sys1").set("mastercoordsystcomp", "3");

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys1");
    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "(sys1.e_t12 + sys1.e_t22)/eta", "0"});

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "(sys1.e_theta2 + sys1.e_phi2)/eta", "0"});
    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "sys2");
    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "(sys2.e_theta2 + sys2.e_phi2)/eta", "0"});

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"0", "ewfd.Ex*(sys2.e_theta2 + sys2.e_phi2)/eta", "0"});

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1").set("Eax", "ewfd.Ex");
    model.component("comp1").variable("var1").rename("Eax", "Etheta");
    model.component("comp1").variable("var1")
         .set("Etheta", "ewfd.Ex*sys2.e_theta1 + ewfd.Ey*sys2.e_theta2 + ewfd.Ez*sys2.e_theta3");
    model.component("comp1").variable("var1")
         .set("Ephi", "ewfd.Ex*sys2.e_phi1 + ewfd.Ey*sys2.e_phi2 + ewfd.Ez*sys2.e_phi3");
    model.component("comp1").variable("var1").set("Etx", "Etheta*sys2.e_theta1 + Ephi*sys2.e_phi1");
    model.component("comp1").variable("var1").set("Ety", "Etheta*sys2.e_theta2 + Ephi*sys2.e_phi2");
    model.component("comp1").variable("var1").set("Etz", "Etheta*sys2.e_theta3 + Ephi*sys2.e_phi3");

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"i*omega_const*epsilon0_const*chi_ee*Etx", "i*omega_const*epsilon0_const*chi_ee*Ety", "i*omega_const*epsilon0_const*chi_ee*Etz"});

    model.param().set("omega_const", "2*pi*f0");
    model.param().descr("omega_const", "rad/s");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"Etx/eta", "Ety/eta", "Etz/eta"});

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmax", "lambda0/8");
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "lambda0/16");
    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lambda0/8");
    model.component("comp1").mesh("mesh1").feature("size1").set("hmin", "lambda0/16");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0/8");
    model.component("comp1").geom("geom1").run("fin");

    model.component("comp1").mesh("mesh1").automatic(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("resolution", "custom");

    model.component("comp1").mesh("mesh1").autoMeshSize(4);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg1").run();
    model.result().create("pg2", "PlotGroup2D");
    model.result("pg2").run();
    model.result("pg2").create("surf1", "Surface");
    model.result().dataset().create("cpl1", "CutPlane");
    model.result().dataset("cpl1").set("data", "dset1");
    model.result("pg2").feature("surf1").set("cutplane", "cpl1");
    model.result("pg2").feature("surf1")
         .set("planefirst", new double[]{-0.0077120968593813646, -0.010282797178094178, 0.0077120968593813646});
    model.result("pg2").feature("surf1")
         .set("planesecond", new double[]{0.0077120968593813646, 0.010282797178094178, -0.0077120968593813646});
    model.result("pg2").feature("surf1").set("planeisinit", true);
    model.result().dataset("cpl1").set("planetype", "general");
    model.result().dataset("cpl1").set("genmethod", "threepoint");
    model.result().dataset("cpl1")
         .set("genpoints", new double[][]{{0, 0, 0}, {0.8000000000000002, -0.6000000000000001, 0}, {0.308696, 0.41159700000000005, 0.8574930000000002}});
    model.result("pg2").feature("surf1").set("cutmode", true);
    model.result("pg2").feature("surf1").set("listen", false);
    model.result("pg2").feature("surf1").set("data", "cpl1");
    model.result("pg2").feature("surf1").set("listen", true);
    model.result("pg2").feature("surf1").set("planedepth", 0.5);
    model.result("pg2").feature("surf1").set("cutmode", false);
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "ewfd.Ex");
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.E");
    model.result("pg2").set("frametype", "mesh");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("cutplane", "cpl1");
    model.result().dataset("cpl1").set("planetype", "general");
    model.result().dataset("cpl1").set("genmethod", "threepoint");
    model.result().dataset("cpl1")
         .set("genpoints", new double[][]{{0, 0, 0}, {0.8000000000000002, -0.6000000000000001, 0}, {0.308696, 0.41159700000000005, 0.8574930000000002}});
    model.result("pg2").feature("surf1").set("cutmode", false);
    model.result().dataset("cpl1").set("genmethod", "pointnormal");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{1, 0, 0});
    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(false);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 2);

    model.label("test.mph");

    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0/4");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0", 2);
    model.component("comp1").geom("geom1").run("fin");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").autoMeshSize(1);

    model.label("test.mph");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"i*omega_const*epsilon0_const*chi_ee*Etx", "i*omega_const*epsilon0_const*chi_ee*Ety", "i*omega_const*epsilon0_const*chi_ee*Etz"});

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{1, 0, 0});
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();

    model.param().set("epsilon", "epsilon0_const");
    model.param().rename("epsilon", "epsilon_0");

    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0/2");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "lambda0*2", 2);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(false);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").set("allowtableupdate", false);
    model.result("pg2").set("xlabel", "");
    model.result("pg2").set("ylabel", "");
    model.result("pg2").feature("surf1").set("rangeunit", "V/m");
    model.result("pg2").feature("surf1").set("rangecolormin", -5.825223056231918);
    model.result("pg2").feature("surf1").set("rangecolormax", 5.931145631667873);
    model.result("pg2").feature("surf1").set("rangecoloractive", "off");
    model.result("pg2").feature("surf1")
         .set("rangeactualminmax", new double[]{-5.825223056231918, 5.931145631667873});
    model.result("pg2").feature("surf1").set("rangeisshared", false);
    model.result("pg2").feature("surf1").set("rangedatamin", -5.825223056231918);
    model.result("pg2").feature("surf1").set("rangedatamax", 5.931145631667873);
    model.result("pg2").feature("surf1").set("rangedataactive", "off");
    model.result("pg2").set("allowtableupdate", true);
    model.result("pg2").set("showlegends", true);
    model.result("pg2").set("showlegendsmaxmin", false);
    model.result("pg2").set("axisactive", true);
    model.result("pg2").set("axisprecision", 4);
    model.result("pg2").set("axisactive", false);
    model.result("pg2").set("xlabelactive", false);
    model.result("pg2").set("ylabelactive", false);
    model.result("pg2").set("view", "auto");
    model.result("pg2").set("legendcolor", "black");
    model.result("pg2").set("axisactive", true);
    model.result("pg2").set("axiscommonexp", true);
    model.result("pg2").set("axisactive", false);
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecoloractive", true);
    model.result("pg2").feature("surf1").set("rangecolormin", -2);
    model.result("pg2").feature("surf1").set("rangecolormax", 2);
    model.result("pg2").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{1, 0, 0});
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "ewfd.Ey");
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ey");
    model.result("pg2").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(false);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "ewfd.Ex");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "ewfd.Ey");
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 0, 1});
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecolormin", -1.5);
    model.result("pg2").feature("surf1").set("rangecolormax", 1.5);

    model.component("comp1").physics("ewfd").feature("scu1").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").study("std1");

    model.study("std1").feature("freq").set("notlistsolnum", 1);
    model.study("std1").feature("freq").set("notsolnum", "auto");
    model.study("std1").feature("freq").set("listsolnum", 1);
    model.study("std1").feature("freq").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "freq");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "freq");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").set("stol", 0.001);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f0"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"Hz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "bicgstab");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "left");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (ewfd)");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1").set("prefun", "sorvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("prefun", "soruvec");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("sv1")
         .set("sorvecdof", new String[]{"comp1_E"});
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "gmres");

    model.result("pg2").run();
    model.result("pg1").run();

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "dDef");

    model.label("test.mph");

    model.component("comp1").variable("var1").set("Eax", "0.5*(up(ewfd.Ex)+down(ewfd.Ex))");
    model.component("comp1").variable("var1").set("Eay", "0.5*(up(ewfd.Ey)+down(ewfd.Ey))");
    model.component("comp1").variable("var1").set("Eaz", "0.5*(up(ewfd.Ez)+down(ewfd.Ez))");
    model.component("comp1").variable("var1")
         .set("Etheta", "Eax*sys2.e_theta1 + ewfd.Ey*sys2.e_theta2 + ewfd.Ez*sys2.e_theta3");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").variable("var1")
         .set("Etheta", "Eax*sys2.e_theta1 + Eay*sys2.e_theta2 + Eaz*sys2.e_theta3");
    model.component("comp1").variable("var1")
         .set("Ephi", "Eax*sys2.e_phi1    + Eay*sys2.e_phi2     +  Eaz*sys2.e_phi3");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Hy");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Hx");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecoloractive", false);
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecoloractive", true);
    model.result("pg2").feature("surf1").set("rangecolormin", -2);
    model.result("pg2").feature("surf1").set("rangecolormax", 2);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.view("view2").set("showgrid", true);

    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecolormin", -3);
    model.result("pg2").feature("surf1").set("rangecolormax", 3);
    model.result("pg2").feature("surf1").set("rangecolormin", -2);
    model.result("pg2").feature("surf1").set("rangecolormax", 2);
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").set("coordinateSystem", "GlobalSystem");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "ewfd.Ex");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(i*k0*z)");
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("wee1").set("minput_temperature_src", "userdef");

    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(-i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(-2*i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(-i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex + E0*exp(-i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("expr", "comp1.ewfd.Ex - E0*exp(-i*k0*z)");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature().duplicate("surf2", "surf1");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf2").set("expr", "comp1.ewfd.Ex");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature().remove("surf2");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").create("surf2", "Surface");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf2").set("data", "cpl1");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf2").set("expr", "comp1.ewfd.Ex");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature().remove("surf2");
    model.result("pg2").run();
    model.result().duplicate("pg3", "pg2");
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").feature("surf1").set("expr", "comp1.ewfd.Ex");
    model.result("pg3").run();
    model.result("pg3").set("data", "cpl1");
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").set("data", "cpl1");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{1, 0, 0});
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").feature("surf1").set("expr", "comp1.ewfd.Ey");
    model.result("pg3").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 0, 1});
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").feature("surf1").set("expr", "comp1.ewfd.Ex");
    model.result("pg3").run();
    model.result("pg3").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(false);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{1, 0, 0});
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.label("test.mph");

    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").variable("var1").set("nEt", "nx*Etx + ny*Ety + nz*Etz");

    model.result("pg3").run();
    model.result().create("pg4", "PlotGroup3D");
    model.result("pg4").run();
    model.result("pg4").create("surf1", "Surface");
    model.result().dataset().create("surf1", "Surface");
    model.result().dataset("surf1").selection().set(6, 7, 8, 9, 10, 11, 12, 13);
    model.result().dataset("surf1").label("Surface_sphere");
    model.result("pg4").run();
    model.result("pg4").set("data", "surf1");
    model.result("pg4").run();
    model.result("pg4").feature("surf1").set("expr", "nx*Etx");
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").feature("surf1").set("expr", "nx*Etx + ny*Ety + nz*Etz");
    model.result("pg4").run();
    model.result("pg4").run();

    model.component("comp1").material("mat1").selection().set(1);

    return model;
  }

  public static Model run4(Model model) {
    model.component("comp1").material("mat1").selection().all();
    model.component("comp1").material("mat2").active(false);

    model.result("pg4").run();

    model.component("comp1").variable("var1").descr("nEt", "sanity check");
    model.component("comp1").variable("var1").set("En", "nx*Eax + ny*Eay + nz*Eaz");
    model.component("comp1").variable("var1").set("Etx_", "Eax - nx*En");
    model.component("comp1").variable("var1").set("Ety_", "Eay - ny*En");
    model.component("comp1").variable("var1").set("Etz_", "Eaz - nz*En");
    model.component("comp1").variable("var1").descr("Etx_", "second way to computetangential vector");
    model.component("comp1").variable("var1").descr("Ety_", "second way to computetangential vector");
    model.component("comp1").variable("var1").descr("Etz_", "second way to computetangential vector");
    model.component("comp1").variable("var1").descr("Ety_", "second way to compute tangential vector");
    model.component("comp1").variable("var1").descr("Etx_", "second way to compute tangential vector");
    model.component("comp1").variable("var1").descr("Etz_", "second way to compute tangential vector");

    model.result("pg4").run();
    model.result("pg3").run();

    model.component("comp1").physics("ewfd").feature("scu1").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature().duplicate("scu2", "scu1");
    model.component("comp1").physics("ewfd").feature("scu1").active(false);
    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"i*omega_const*epsilon0_const*chi_ee*Etx_", "i*omega_const*epsilon0_const*chi_ee*Ety_", "i*omega_const*epsilon0_const*chi_ee*Etz_"});

    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result().duplicate("pg5", "pg4");
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "nx*Etx_ + ny*Ety_ + nz*Etz_");
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.component("comp1").physics("ewfd").feature("scu1")
         .set("Js0", new String[]{"i*omega_const*epsilon0_const*chi_ee*Etx", "i*omega_const*epsilon0_const*chi_ee*Ety", "i*omega_const*epsilon0_const*chi_ee*Etz"});
    model.component("comp1").physics("ewfd").feature("scu2")
         .set("Js0", new String[]{"i*omega_const*epsilon0_const*chi_ee*Etx_", "i*omega_const*epsilon0_const*chi_ee*Ety_", "i*omega_const*epsilon0_const*chi_ee*Etz_"});

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.component("comp1").physics("ewfd").feature("scu2").active(false);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").feature("scu2").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg5").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").create("smcd1", "SurfaceMagneticCurrentDensity", 2);
    model.component("comp1").physics("ewfd").feature("smcd1").selection().set(6, 7, 8, 9, 10, 11, 12, 13);

    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg4").run();

    model.component("comp1").variable("var1").remove("nEt");
    model.component("comp1").variable("var1").set("Hax", "0.5*(up(ewfd.Hx)+down(ewfd.Hx))");
    model.component("comp1").variable("var1").set("Hay", "0.5*(up(ewfd.Hy)+down(ewfd.Hy))");
    model.component("comp1").variable("var1").set("Haz", "0.5*(up(ewfd.Hz)+down(ewfd.Hz))");
    model.component("comp1").variable("var1").set("Hn", "nx*Hax + ny*Hay + nz*Haz");
    model.component("comp1").variable("var1").set("Htx", "Hax - nx*Hn");
    model.component("comp1").variable("var1").set("Hty", "Hay - ny*Hn");
    model.component("comp1").variable("var1").set("Htz", "Haz - nz*Hn");
    model.component("comp1").variable("var1").rename("Htx", "Htx_");
    model.component("comp1").variable("var1").rename("Hty", "Hty_");
    model.component("comp1").variable("var1").rename("Htz", "Htz_");

    model.param().set("chi_mm", "0.1");
    model.param().descr("chi_mm", "Magnetic susceptibility");
    model.param().set("chi_em", "0.0");
    model.param().set("chi_me", "0.0");
    model.param().set("sheetE", "1");
    model.param().set("sheetM", "1");
    model.param().descr("sheetE", "on/off sheet");
    model.param().descr("sheetM", "on/off sheet");

    model.component("comp1").physics("ewfd").feature("scu2")
         .set("Js0", new String[]{"sheetE * ( i*omega_const*epsilon0_const*chi_ee*Etx_ + i*k0*chi_em*Htx_ )", "sheetE * ( i*omega_const*epsilon0_const*chi_ee*Ety_ + i*k0*chi_em*Hty_ )", "sheetE * ( i*omega_const*epsilon0_const*chi_ee*Etz_ + i*k0*chi_em*Htz_ )"});
    model.component("comp1").physics("ewfd").feature("smcd1")
         .set("Jms0", new String[]{"sheetM * ( -i*omega_const*mu0_const*chi_mm*Htx_ - i*k0*chi_me*Etx_ )", "sheetM * ( -i*omega_const*mu0_const*chi_mm*Hty_ - i*k0*chi_me*Ety_ )", "sheetM * ( -i*omega_const*mu0_const*chi_mm*Htz_ - i*k0*chi_me*Etz_ )"});

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "dDef");

    model.result("pg5").run();

    model.param().set("chi_mm", "0.01");
    model.param().set("chi_ee", "0.01");
    model.param().set("sheetM", "0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.param().set("chi_mm", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.param().set("chi_ee", "0.1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.param().set("chi_mm", "0.1");
    model.param().set("sheetE", "0");
    model.param().set("sheetM", "1");
    model.param().set("chi_mm", "0.0");
    model.param().set("chi_em", "0.1");
    model.param().set("chi_me", "0.1");
    model.param().set("chi_em", "0.0");

    model.component("comp1").physics("ewfd").feature("scu2").active(false);
    model.component("comp1").physics("ewfd").field("electricfield").field("H");
    model.component("comp1").physics("ewfd").field("electricfield").field("E");

    model.param().set("chi_me", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.param().set("chi_me", "0.0");
    model.param().set("chi_mm", "1e-4");

    model.sol("sol1").feature("s1").feature("fc1").set("dtech", "auto");
    model.sol("sol1").feature("s1").feature("dDef").set("linsolver", "pardiso");

    model.param().set("chi_mm", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("chi_me", "0.00001");

    model.component("comp1").variable("var1").set("Eax", "ewfd.Ex");
    model.component("comp1").variable("var1").set("Eay", "ewfd.Ey");
    model.component("comp1").variable("var1").set("Eaz", "ewfd.Ez");
    model.component("comp1").variable("var1").set("Hax", "ewfd.Hx");
    model.component("comp1").variable("var1").set("Hay", "ewfd.Hy");
    model.component("comp1").variable("var1").set("Haz", "ewfd.Hz");

    model.param().set("sheetM", "0.0001");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.param().set("chi_me", "0.0");
    model.param().set("chi_mm", "0.0001");
    model.param().set("f0", "20[GHz]");

    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg1").run();

    model.param().set("chi_mm", "0.0");
    model.param().set("chi_me", "0.0001");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("chi_me", "0.001");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("chi_me", "0.01");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("f0", "10[GHz]");

    model.label("test.mph");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.param().set("chi_me", "0.01");
    model.param().set("sheetM", "0.001");
    model.param().set("chi_me", "0.0001");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("chi_me", "0.001");
    model.param().set("sheetM", "0.01");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("sheetM", "0.1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg2").run();

    model.param().set("sheetM", "1");
    model.param().set("chi_me", "0.1*(1+i*1e-3)");

    model.component("comp1").physics("ewfd").field("electricfield").field("H");
    model.component("comp1").physics("ewfd").field("electricfield").component(1, "Hx");
    model.component("comp1").physics("ewfd").field("electricfield").component(2, "Hy");
    model.component("comp1").physics("ewfd").field("electricfield").component(3, "Hz");
    model.component("comp1").physics("ewfd").prop("EquationForm").setIndex("form", "Frequency", 0);
    model.component("comp1").physics("ewfd").prop("EquationForm").setIndex("freq_src", "userdef", 0);
    model.component("comp1").physics("ewfd").prop("EquationForm").setIndex("freq_src", "fromSolver", 0);
    model.component("comp1").physics("ewfd").prop("EquationForm").setIndex("form", "BoundaryModeAnalysis", 0);
    model.component("comp1").physics("ewfd").prop("EquationForm").setIndex("form", "Automatic", 0);
    model.component("comp1").physics("ewfd").prop("BackgroundField").set("SolveFor", "fullField");
    model.component("comp1").physics("ewfd").prop("ShapeProperty").set("order_electricfield", 2);
    model.component("comp1").physics("ewfd").field("electricfield").field("E");
    model.component("comp1").physics("ewfd").field("electricfield").component(1, "Ex");
    model.component("comp1").physics("ewfd").field("electricfield").component(2, "Ey");
    model.component("comp1").physics("ewfd").field("electricfield").component(3, "Ez");

    model.param().set("chi_me", "0.1");

    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").autoMeshSize(4);
    model.component("comp1").mesh("mesh1").autoMeshSize(1);

    model.sol("sol1").feature("s1").feature("fc1").set("plot", false);

    model.param().set("chi_me", "0.0");
    model.param().set("chi_mm", "0.000000001");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.param().set("chi_mm", "0.00001");

    model.label("test.mph");

    model.component("comp1").mesh("mesh1").autoMeshSize(6);
    model.component("comp1").mesh("mesh1").autoMeshSize(9);
    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.7990E-3");
    model.component("comp1").mesh("mesh1").automatic(true);
    model.component("comp1").mesh("mesh1").autoMeshSize(1);

    model.component("comp1").physics("ewfd").feature("scu2").active(true);

    model.param().set("chi_mm", "0.1");
    model.param().set("chi_em", "0.1");
    model.param().set("chi_me", "0.1");
    model.param().set("sheetE", "0");
    model.param().set("sheetM", "0.00000000001");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").variable("var1").set("Eax", "0.5*(up(ewfd.Ex) + down(ewfd.Ex))");
    model.component("comp1").variable("var1").set("Eay", "0.5*(up(ewfd.Ex) + down(ewfd.Ex))");
    model.component("comp1").variable("var1").set("Eaz", "0.5*(up(ewfd.Ex) + down(ewfd.Ex))");
    model.component("comp1").variable("var1").set("Eay", "0.5*(up(ewfd.Ey) + down(ewfd.Ey))");
    model.component("comp1").variable("var1").set("Eaz", "0.5*(up(ewfd.Ez) + down(ewfd.Ez))");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("sheetM", "0.00000001");

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.param().set("sheetM", "0.00001");

    model.sol("sol1").feature("s1").feature("aDef").set("convinfo", "detailed");
    model.sol("sol1").feature("s1").feature("aDef").set("autorescale", true);

    model.study("std1").create("freq2", "Frequency");
    model.study("std1").feature().remove("freq2");
    model.study("std1").feature("freq").set("useadvanceddisable", true);

    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2000E-3");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 1);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2000E-2");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.param().set("sheetM", "1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2000E-2/2");

    model.component("comp1").physics("ewfd").prop("MeshControl").set("SizeControlParameter", "FromStudy");
    model.component("comp1").physics("ewfd").prop("ShapeProperty").set("order_electricfield", 1);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmax", ".1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2000E-2/3");
    model.component("comp1").mesh("mesh1").feature("size").set("custom", false);
    model.component("comp1").mesh("mesh1").create("bl1", "BndLayer");
    model.component("comp1").mesh("mesh1").feature("bl1").create("blp", "BndLayerProp");
    model.component("comp1").mesh("mesh1").feature("bl1").selection().geom("geom1", 3);
    model.component("comp1").mesh("mesh1").feature("bl1").selection().all();
    model.component("comp1").mesh("mesh1").feature("bl1").selection().named("geom1_sph1_dom");
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").selection().all();
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").selection().set(6, 7, 8, 9, 10, 11, 12, 13);
    model.component("comp1").mesh("mesh1").run("bl1");
    model.component("comp1").mesh("mesh1").feature("bl1").selection().set();
    model.component("comp1").mesh("mesh1").run("bl1");
    model.component("comp1").mesh("mesh1").feature("bl1").selection().named("geom1_sph1_dom");
    model.component("comp1").mesh("mesh1").feature("size").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-3");
    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.1);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").physics("ewfd").prop("ShapeProperty").set("order_electricfield", 2);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").set("blstretch", 1.2);
    model.component("comp1").mesh("mesh1").feature("bl1").selection().all();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.01);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2/2");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();

    model.param().set("sheetE", "1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2/3");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("bl1").selection().set(2);
    model.component("comp1").mesh("mesh1").feature("bl1").selection().geom("geom1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("multiplanexmethod", "coord");
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecoloractive", false);
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").feature("surf1").set("rangecoloractive", false);

    model.param().set("sheetE", "0");

    model.label("test.mph");

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").set("blnlayers", 4);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").feature("surf1").set("rangecoloractive", true);
    model.result("pg2").feature("surf1").set("rangecolormin", -2);
    model.result("pg2").feature("surf1").set("rangecolormax", 2);

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").selection().all();
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp").active(false);
    model.component("comp1").mesh("mesh1").feature("bl1").feature().remove("blp");
    model.component("comp1").mesh("mesh1").create("ada1", "Adapt");
    model.component("comp1").mesh("mesh1").feature("ada1").set("method", "modify");
    model.component("comp1").mesh("mesh1").feature("ada1").active(false);
    model.component("comp1").mesh("mesh1").feature().remove("ada1");
    model.component("comp1").mesh("mesh1").create("ref1", "Refine");
    model.component("comp1").mesh("mesh1").feature("ref1").selection().geom("geom1", 2);
    model.component("comp1").mesh("mesh1").feature("ref1").selection().set(6, 7, 8, 9, 10, 11, 12, 13);
    model.component("comp1").mesh("mesh1").run("ref1");
    model.component("comp1").mesh("mesh1").feature("bl1").active(false);
    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2/2");
    model.component("comp1").mesh("mesh1").feature("ref1").active(false);
    model.component("comp1").mesh("mesh1").feature("bl1").active(true);
    model.component("comp1").mesh("mesh1").feature("bl1").create("blp1", "BndLayerProp");
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 3);
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").selection().all();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("ref1").active(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("ref1").active(false);
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 2);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("sheetE", "1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").feature("surf1").set("rangecoloractive", true);
    model.result("pg3").feature("surf1").set("rangecolormin", -2);
    model.result("pg3").feature("surf1").set("rangecolormax", 2);

    model.param().set("chi_em", "0.0");
    model.param().set("chi_me", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmin", "1.2E-2/3");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.001);
    model.component("comp1").mesh("mesh1").run("ftet1");
    model.component("comp1").mesh("mesh1").feature("size").set("custom", false);
    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 5);
    model.component("comp1").mesh("mesh1").run("size");
    model.component("comp1").mesh("mesh1").run();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("chi_em", "0.01");
    model.param().set("chi_me", "0.01");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("chi_em", "0.1");
    model.param().set("chi_me", "0.1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("chi_em", "0.0");
    model.param().set("chi_me", "0.0");
    model.param().set("chi_mm", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.param().set("chi_mm", "0.1");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").run();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.label("test.mph");

    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.param().set("chi_mm", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("chi_mm", "0.1");
    model.param().set("sheetE", "0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);
    model.component("comp1").mesh("mesh1").feature("bl1").active(false);
    model.component("comp1").mesh("mesh1").feature("bl1").active(true);

    model.study("std1").feature("freq").set("preusesol", "no");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 3);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 4);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();

    model.param().set("sheetE", "1");

    model.label("test.mph");

    model.component("comp1").geom("geom1").feature("sph1").set("r", "lambda0");

    model.param().set("radius", "lambda0");
    model.param().set("f0", "1[GHz]");

    model.component("comp1").geom("geom1").feature("sph1").set("r", "radius");
    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "2*radius", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "2*radius", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "2*radius", 2);
    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").run();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg5").run();

    model.param().set("chi_mm", "0.0");

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg1").run();

    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "4*radius", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "4*radius", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "4*radius", 2);
    model.component("comp1").geom("geom1").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 5);
    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 1);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();

    model.label("test.mph");

    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result().dataset("cpl1").set("genpnvec", new int[]{0, 1, 0});
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").physics("ewfd").feature("sctr1").set("kdir", new String[]{"0", "0", "k0"});

    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 2);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("size").set("hauto", 3);
    model.component("comp1").mesh("mesh1").run();

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").mesh("mesh1").feature("bl1").feature("blp1").set("blnlayers", 3);

    model.label("test.mph");

    model.component("comp1").mesh("mesh1").automatic(true);

    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.param().set("chi_mm", "0.1");

    model.label("test.mph");

    return model;
  }

  public static void main(String[] args) {
    Model model = run();
    model = run2(model);
    model = run3(model);
    run4(model);
  }

}
