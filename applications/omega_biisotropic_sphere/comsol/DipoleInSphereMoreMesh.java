/*
 * DipoleInSphereMoreMesh.java
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model exported on Sep 14 2026, 12:31 by COMSOL 6.1.0.357. */
public class DipoleInSphereMoreMesh {

  public static Model run() {
    Model model = ModelUtil.create("Model");

    model.modelPath("C:\\Users\\German\\Documents\\COMSOL");

    model.component().create("comp1", true);

    model.component("comp1").geom().create("geom1", 3);

    model.component("comp1").mesh().create("mesh1");

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");

    model.study().create("std1");
    model.study("std1").create("fdrfam", "FrequencyDomainRFAdaptiveMesh");
    model.study("std1").feature("fdrfam").set("solnum", "auto");
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("ngenAUX", "1");
    model.study("std1").feature("fdrfam").set("goalngenAUX", "1");
    model.study("std1").feature("fdrfam").set("ngenAUX", "1");
    model.study("std1").feature("fdrfam").set("goalngenAUX", "1");
    model.study("std1").feature("fdrfam").setSolveFor("/physics/emw", true);

    model.component("comp1").geom("geom1").run();

    model.param().set("a", "l");
    model.param().rename("a", "lam");
    model.param().set("lam", "3e8[m/s]/f");
    model.param().set("f", "10[GHz]");
    model.param().set("a", "0.7*lam");
    model.param().set("er", "7");
    model.param().set("Xem", "0.38123");
    model.param().set("Yee", "-0.0036679[S/m]");
    model.param().set("Zmm", "-184.0435[ohm/m]");
    model.param().set("Yee", "-0.0036679[S]");
    model.param().set("Zmm", "-184.0435[ohm]");

    model.component("comp1").geom("geom1").create("blk1", "Block");

    model.param().set("pml", "lam/4");

    model.component("comp1").geom("geom1").feature("blk1").set("size", new String[]{"2*a+4*pml", "2*a+4*pml", "1"});
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "2*a+4*pml", 2);
    model.component("comp1").geom("geom1").feature("blk1").set("base", "center");
    model.component("comp1").geom("geom1").feature("blk1").setIndex("layer", "pml", 0);
    model.component("comp1").geom("geom1").feature("blk1").set("layerleft", true);
    model.component("comp1").geom("geom1").feature("blk1").set("layerright", true);
    model.component("comp1").geom("geom1").feature("blk1").set("layerfront", true);
    model.component("comp1").geom("geom1").feature("blk1").set("layerback", true);
    model.component("comp1").geom("geom1").feature("blk1").set("layertop", true);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run();

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
    model.component("comp1").material("mat1").propertyGroup().create("idealGas", "Ideal gas");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func().create("Cp", "Piecewise");
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
    model.component("comp1").material("mat1").propertyGroup("def").func("rho").set("fununit", "kg/m^3");
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("argunit", new String[]{"Pa", "K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("rho")
         .set("plotargs", new String[][]{{"pA", "101325", "101325"}, {"T", "273.15", "293.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("def").func("k")
         .set("pieces", new String[][]{{"200.0", "1600.0", "-0.00227583562+1.15480022E-4*T^1-7.90252856E-8*T^2+4.11702505E-11*T^3-7.43864331E-15*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("def").func("k").set("fununit", "W/(m*K)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("expr", "sqrt(1.4*R_const[K*mol/J]/0.02897*T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("fununit", "m/s");
    model.component("comp1").material("mat1").propertyGroup("def").func("cs").set("argunit", new String[]{"K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("cs")
         .set("plotargs", new String[][]{{"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("funcname", "alpha_p");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("expr", "-1/rho(pA,T)*d(rho(pA,T),T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("args", new String[]{"pA", "T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1").set("fununit", "1/K");
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("argunit", new String[]{"Pa", "K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an1")
         .set("plotargs", new String[][]{{"pA", "101325", "101325"}, {"T", "273.15", "373.15"}});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("funcname", "muB");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("expr", "0.6*eta(T)");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("args", new String[]{"T"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("fununit", "Pa*s");
    model.component("comp1").material("mat1").propertyGroup("def").func("an2").set("argunit", new String[]{"K"});
    model.component("comp1").material("mat1").propertyGroup("def").func("an2")
         .set("plotargs", new String[][]{{"T", "200", "1600"}});
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "");
    model.component("comp1").material("mat1").propertyGroup("def")
         .set("thermalexpansioncoefficient", new String[]{"alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)", "0", "0", "0", "alpha_p(pA,T)"});
    model.component("comp1").material("mat1").propertyGroup("def").set("molarmass", "0.02897[kg/mol]");
    model.component("comp1").material("mat1").propertyGroup("def").set("bulkviscosity", "muB(T)");
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
    model.component("comp1").material("mat1").propertyGroup("def").addInput("temperature");
    model.component("comp1").material("mat1").propertyGroup("def").addInput("pressure");
    model.component("comp1").material("mat1").propertyGroup("RefractiveIndex")
         .set("n", new String[]{"1", "0", "0", "0", "1", "0", "0", "0", "1"});
    model.component("comp1").material("mat1").propertyGroup("NonlinearModel").set("BA", "(def.gamma+1)/2");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").label("Piecewise 2");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("arg", "T");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp")
         .set("pieces", new String[][]{{"200.0", "1600.0", "1047.63657-0.372589265*T^1+9.45304214E-4*T^2-6.02409443E-7*T^3+1.2858961E-10*T^4"}});
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("argunit", "K");
    model.component("comp1").material("mat1").propertyGroup("idealGas").func("Cp").set("fununit", "J/(kg*K)");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("Rs", "R_const/Mn");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("heatcapacity", "Cp(T)");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("ratioofspecificheat", "1.4");
    model.component("comp1").material("mat1").propertyGroup("idealGas").set("molarmass", "0.02897");
    model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("temperature");
    model.component("comp1").material("mat1").propertyGroup("idealGas").addInput("pressure");
    model.component("comp1").material("mat1").materialType("nonSolid");
    model.component("comp1").material("mat1").set("family", "air");

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection()
         .set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27);

    model.component("comp1").view("view1").hideObjects().create("hide1");
    model.component("comp1").view("view1").hideObjects("hide1").init(3);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 13);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 4);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 7);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 16);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 22);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 25);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 23);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 17);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 26);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 18);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 24);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 27);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 15);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 9);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 8);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 6);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 5);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 3);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 2);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 1);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 10);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 19);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 11);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 20);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 12);
    model.component("comp1").view("view1").hideObjects("hide1").add("blk1", 21);
    model.component("comp1").view("view1").hideObjects().create("hide2");
    model.component("comp1").view("view1").hideObjects("hide2").init(2);
    model.component("comp1").view("view1").hideObjects("hide2").add("blk1", 80);
    model.component("comp1").view("view1").hideObjects("hide2").add("blk1", 58);
    model.component("comp1").view("view1").hideObjects("hide2").add("blk1", 52);

    model.component("comp1").geom("geom1").run("blk1");
    model.component("comp1").geom("geom1").create("sph1", "Sphere");
    model.component("comp1").geom("geom1").feature("sph1").set("r", "a");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run();

    model.component("comp1").view("view1").set("hidestatus", "ignore");

    model.component("comp1").physics("emw").create("sctr1", "Scattering", 2);
    model.component("comp1").physics("emw").feature("sctr1").selection()
         .set(1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 20, 21, 23, 24, 27, 30, 31, 32, 33, 35, 36, 38, 41, 43, 46, 53, 56, 63, 64, 65, 66, 76, 77, 79, 82, 84, 87, 94, 97, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116);

    model.component("comp1").view("view1").set("hidestatus", "hide");

    model.component("comp1").material().create("mat2", "Common");

    model.component("comp1").view("view1").set("renderwireframe", true);
    model.component("comp1").view("view1").set("transparency", false);
    model.component("comp1").view("view1").set("renderwireframe", false);
    model.component("comp1").view("view1").set("transparency", false);

    model.component("comp1").material("mat2").selection().set(19);
    model.component("comp1").material("mat2").propertyGroup("def").set("relpermittivity", new String[]{"er"});
    model.component("comp1").material("mat2").propertyGroup("def").set("relpermeability", new String[]{"1"});
    model.component("comp1").material("mat2").propertyGroup("def").set("electricconductivity", new String[]{"0"});

    model.component("comp1").physics("emw").create("scu1", "SurfaceCurrent", 2);
    model.component("comp1").physics("emw").feature("scu1").selection().set(67, 68, 69, 70, 71, 72, 73, 74);

    model.component("comp1").coordSystem().create("sys2", "Spherical");

    model.component("comp1").physics("emw").feature("scu1").set("coordinateSystem", "sys2");
    model.component("comp1").physics("emw").feature("scu1").active(false);

    model.component("comp1").geom("geom1").run("sph1");
    model.component("comp1").geom("geom1").create("pt1", "Point");
    model.component("comp1").geom("geom1").feature("pt1").set("selresult", true);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run();

    model.component("comp1").physics("emw").create("epd1", "ElectricPointDipole", 0);

    model.component("comp1").view("view1").set("renderwireframe", true);

    model.component("comp1").physics("emw").feature("epd1").selection().set(36);
    model.component("comp1").physics("emw").feature("epd1").set("normpI", 1);
    model.component("comp1").physics("emw").feature("epd1").set("enpI", new int[]{0, 0, 1});

    model.study("std1").feature("fdrfam").set("plist", "f");

    model.sol().create("sol1");

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").set("data", "dset2");
    model.result("pg1").set("defaultPlotID", "ElectromagneticWaves/phys1/pdef1/pcond1/pg1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").label("Multislice");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result("pg1").feature("mslc1").feature().create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");

    model.component("comp1").mesh("mesh1").run("dis1");

    model.result().remove("pg1");

    model.component("comp1").physics("emw").create("ffd1", "FarFieldDomain", 3);

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "Default");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result().create("pg1", "PlotGroup3D");
    model.result("pg1").label("Electric Field (emw)");
    model.result("pg1").set("frametype", "spatial");
    model.result("pg1").set("showlegendsmaxmin", true);
    model.result("pg1").set("data", "dset2");
    model.result("pg1").set("defaultPlotID", "ElectromagneticWaves/phys1/pdef1/pcond1/pg1");
    model.result("pg1").feature().create("mslc1", "Multislice");
    model.result("pg1").feature("mslc1").label("Multislice");
    model.result("pg1").feature("mslc1").set("smooth", "internal");
    model.result("pg1").feature("mslc1").set("data", "parent");
    model.result("pg1").feature("mslc1").feature().create("filt1", "Filter");
    model.result("pg1").feature("mslc1").feature("filt1").set("expr", "!isScalingSystemDomain");
    model.result().create("pg2", "PolarGroup");
    model.result("pg2").label("2D Far Field (emw)");
    model.result("pg2").set("data", "dset2");
    model.result("pg2").create("rp1", "RadiationPattern");
    model.result("pg2").feature("rp1").set("legend", "on");
    model.result("pg2").feature("rp1").set("phidisc", "180");
    model.result("pg2").feature("rp1").set("expr", new String[]{"emw.normEfar"});
    model.result().create("pg3", "PlotGroup3D");
    model.result("pg3").label("3D Far Field (emw)");
    model.result("pg3").set("data", "dset2");
    model.result("pg3").set("view", "new");
    model.result("pg3").set("edges", "off");
    model.result("pg3").set("showlegendsmaxmin", true);
    model.result("pg3").create("rp1", "RadiationPattern");
    model.result("pg3").feature("rp1").set("expr", new String[]{"emw.normEfar"});
    model.result("pg3").feature("rp1").set("colorexpr", new String[]{"emw.normEfar"});
    model.result("pg3").feature("rp1").set("useradiusascolor", true);
    model.result("pg3").feature("rp1").set("directivityexpr", new String[]{"emw.normEfar^2"});
    model.result("pg3").feature("rp1").set("thetadisc", "45");
    model.result("pg3").feature("rp1").set("phidisc", "45");
    model.result("pg3").feature("rp1").set("directivity", "on");
    model.result("pg3").feature("rp1").set("colortable", "RainbowLight");

    model.sol("sol1").runAll();

    model.result("pg1").set("data", "dset2");
    model.result("pg1").run();
    model.result("pg3").run();
    model.result("pg1").run();
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").view("view1").set("hidestatus", "ignore");
    model.component("comp1").view("view1").set("transparency", true);

    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", true);
    model.result("pg1").feature("mslc1").set("rangecolormin", 45438.2698562296);
    model.result("pg1").feature("mslc1").set("rangecolormax", "4.0E6");

    model.component("comp1").view("view1").set("transparency", false);

    model.result("pg1").feature("mslc1").set("rangecolormin", "4e2");
    model.result("pg1").feature("mslc1").set("rangecolormax", "4.0E6");
    model.result("pg2").run();
    model.result("pg3").run();

    model.component("comp1").view("view1").set("transparency", true);

    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").run();
    model.result().create("pg4", "PlotGroup1D");
    model.result("pg4").run();
    model.result("pg4").create("rp1", "RadiationPattern");
    model.result("pg4").feature("rp1").set("markerpos", "datapoints");
    model.result("pg4").feature("rp1").set("linewidth", "preference");
    model.result("pg4").feature("rp1").set("phidisc", 361);
    model.result("pg4").feature("rp1").set("normal", new int[]{1, 0, 0});
    model.result("pg4").feature("rp1").set("refdir", new int[]{0, 0, -1});
    model.result("pg4").feature("rp1").set("normal", new int[]{0, -1, 0});
    model.result("pg4").run();

    model.study("std1").create("freq", "Frequency");
    model.study("std1").feature("fdrfam").active(false);
    model.study("std1").feature("freq").set("plist", "f");
    model.study("std1").feature("freq").setEntry("mesh", "geom1", "mesh1");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");

    return model;
  }

  public static Model run2(Model model) {
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").feature("rp1").set("expr", "emw.normdBEfar");
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg3").set("data", "dset1");
    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg4").feature().duplicate("rp2", "rp1");
    model.result("pg4").run();
    model.result("pg4").feature("rp2").set("normal", new int[]{0, 0, 1});
    model.result("pg4").feature("rp2").set("refdir", new int[]{1, 0, 0});
    model.result("pg4").run();

    model.component("comp1").physics("emw").feature("sctr1").set("WaveType", "SphericalWave");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg2").feature("rp1").set("expr", "emw.normdBEfar");
    model.result("pg2").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();

    model.component("comp1").mesh("mesh1").autoMeshSize(1);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg3").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.Ex");
    model.result("pg1").feature("mslc1").set("descr", "Electric field, x-component");
    model.result("pg1").feature("mslc1").set("expr", "real(emw.Ex)");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", false);
    model.result("pg1").run();
    model.result("pg1").set("data", "dset1");
    model.result("pg1").run();

    model.component("comp1").view("view1").set("transparency", false);

    model.component("comp1").physics("emw").feature("scu1").active(true);

    model.component("comp1").view("view1").set("transparency", true);
    model.component("comp1").view("view1").set("hidestatus", "hide");

    model.result("pg1").run();

    model.component("comp1").physics("emw").feature("scu1").set("Js0", new int[]{0, 1, 0});
    model.component("comp1").physics("emw").create("smcd1", "SurfaceMagneticCurrentDensity", 2);
    model.component("comp1").physics("emw").feature("scu1").set("coordinateSystem", "GlobalSystem");
    model.component("comp1").physics("emw").feature("smcd1").set("Jms0", new int[]{1, 0, 0});
    model.component("comp1").physics("emw").feature("smcd1").selection().set(67, 68, 69, 70, 71, 72, 73, 74);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg3").run();
    model.result("pg2").run();
    model.result("pg1").run();
    model.result("pg4").run();
    model.result("pg1").run();

    model.study("std1").feature("freq").active(false);
    model.study("std1").feature("fdrfam").active(true);
    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg2").run();
    model.result("pg2").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg3").run();
    model.result("pg1").run();

    model.component("comp1").physics("emw").feature("epd1").active(false);

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg2").run();
    model.result("pg1").run();

    model.component("comp1").physics("emw").feature("scu1").set("coordinateSystem", "sys2");

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg2").run();
    model.result("pg1").run();
    model.result().create("pg5", "PlotGroup2D");
    model.result("pg5").run();
    model.result("pg5").create("surf1", "Surface");
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").set("data", "dset2");
    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result().remove("pg5");
    model.result().create("pg5", "PlotGroup3D");
    model.result("pg5").run();
    model.result("pg5").create("surf1", "Surface");
    model.result("pg5").run();
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(70, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(69, 70, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(69, 70, 73, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(69, 70, 71, 73, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(69, 70, 71, 72, 73, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(68, 69, 70, 71, 72, 73, 74);
    model.result("pg5").selection().geom("geom1", 2);
    model.result("pg5").selection().set(67, 68, 69, 70, 71, 72, 73, 74);
    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1").set("coordinateSystem", "GlobalSystem");
    model.component("comp1").physics("emw").feature("epd1").active(true);

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(Yee/2)*emw.Eupx+(Yee/2)*emw.Edownx", "(Yee/2)*emw.Eupy+(Yee/2)*emw.Edowny", "(Yee/2)*emw.Eupz+(Yee/2)*emw.Edownz"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupz,y)-d(emw.Eupy,z))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edownz,y)-d(emw.Edowny,z))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupx,z)-d(emw.Eupz,x))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edownx,z)-d(emw.Edownz,x))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupy,x)-d(emw.Eupx,y))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edowny,x)-d(emw.Edownx,y))"});

    model.param().set("w", "2*pi*f");
    model.param().set("mu0", "4*pi*10^-7[H/m]");
    model.param().set("w", "2*pi*f[rad]");

    model.component("comp1").physics("emw").feature().move("smcd1", 6);

    model.study("std1").feature("fdrfam").set("saveres", false);

    return model;
  }

  public static Model run3(Model model) {
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").set("data", "dset1");
    model.result("pg1").run();
    model.result("pg1").set("data", "dset2");
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("expr", "emw.normE");
    model.result("pg1").feature("mslc1").set("descr", "Electric field norm");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", true);
    model.result("pg1").feature("mslc1").set("rangecolormin", 4.0E9);
    model.result("pg1").feature("mslc1").set("rangecolormax", 6.0E9);
    model.result("pg1").feature("mslc1").set("rangecolormin", 43028.6019571339);
    model.result("pg1").feature("mslc1").set("rangecolormax", 4.0E9);
    model.result("pg1").feature("mslc1").set("expr", "real(emw.Ex)");
    model.result("pg1").run();
    model.result("pg1").feature("mslc1").set("rangecoloractive", true);
    model.result("pg1").feature("mslc1").set("rangecolormin", "-1e5");
    model.result("pg1").feature("mslc1").set("rangecolormax", "1e5");
    model.result("pg1").run();
    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(Yee/2)*emw.tEupx+(Yee/2)*emw.tEdownx", "(Yee/2)*emw.tEupy+(Yee/2)*emw.tEdowny", "(Yee/2)*emw.tEupz+(Yee/2)*emw.tEdownz"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEupz,y)-d(emw.tEupy,z))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEdownz,y)-d(emw.tEdowny,z))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEupx,z)-d(emw.tEupz,x))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEdownx,z)-d(emw.tEdownz,x))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEupy,x)-d(emw.tEupx,y))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.tEdowny,x)-d(emw.tEdownx,y))"});

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(Yee/2)*emw.Eupx+(Yee/2)*emw.Edownx", "(Yee/2)*emw.Eupy+(Yee/2)*emw.Edowny", "(Yee/2)*emw.Eupz+(Yee/2)*emw.Edownz"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupz,y)-d(emw.Eupy,z))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edownz,y)-d(emw.Edowny,z))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupx,z)-d(emw.Eupz,x))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edownx,z)-d(emw.Edownz,x))", "(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Eupy,x)-d(emw.Eupx,y))+(Zmm/2)*(-1/(j*w*mu0))*(d(emw.Edowny,x)-d(emw.Edownx,y))"});
    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(((Yee/2)*(emw.Eupx+emw.Edownx))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z)))))*cos(T(x,y,z))*cos(P(x,y,z))", "(((Yee/2)*(emw.Eupy+emw.Edowny))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,z)-d(emw.Eupy,x))+(d(emw.Edownz,z)-d(emw.Edowny,x)))))*cos(T(x,y,z))*sin(P(x,y,z))", "(((Yee/2)*(emw.Eupz+emw.Edownz))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,x)-d(emw.Eupy,y))+(d(emw.Edownz,x)-d(emw.Edowny,y)))))*(-sin(T(x,y,z)))"});

    model.component("comp1").func().create("an1", "Analytic");
    model.component("comp1").func("an1").set("funcname", "T");
    model.component("comp1").func("an1").set("expr", "acos(z/R(x,y,z))");
    model.component("comp1").func("an1").set("args", "x,y,z");
    model.component("comp1").func("an1").set("fununit", "rad");
    model.component("comp1").func("an1").setIndex("argunit", "m", 0);
    model.component("comp1").func("an1").setIndex("argunit", "m", 1);
    model.component("comp1").func("an1").setIndex("argunit", "m", 2);
    model.component("comp1").func().create("an2", "Analytic");
    model.component("comp1").func("an2").set("funcname", "R");
    model.component("comp1").func("an2").set("expr", "sqrt(x^2+y^2+z^2)");
    model.component("comp1").func("an2").set("args", "x,y,z");
    model.component("comp1").func("an2").set("fununit", "m");
    model.component("comp1").func("an2").setIndex("argunit", "m", 0);
    model.component("comp1").func("an2").setIndex("argunit", "m", 1);
    model.component("comp1").func("an2").setIndex("argunit", "m", 2);
    model.component("comp1").func().create("an3", "Analytic");
    model.component("comp1").func("an3").set("funcname", "P");
    model.component("comp1").func("an3").set("expr", "atan2(y,x)");
    model.component("comp1").func("an3").set("args", "x,y");
    model.component("comp1").func("an3").set("fununit", "rad");
    model.component("comp1").func("an3").setIndex("argunit", "m", 0);
    model.component("comp1").func("an3").setIndex("argunit", "m", 1);

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(((Yee/2)*(emw.Eupx+emw.Edownx))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z)))))*cos(T(x,y,z))*cos(P(x,y,z))", "(((Yee/2)*(emw.Eupy+emw.Edowny))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,z)-d(emw.Eupy,x))+(d(emw.Edownz,z)-d(emw.Edowny,x)))))*cos(T(x,y,z))*sin(P(x,y,z))", "(((Yee/2)*(emw.Eupz+emw.Edownz))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,x)-d(emw.Eupy,y))+(d(emw.Edownz,x)-d(emw.Edowny,y)))))*(-sin(T(x,y,z)))"});

    model.component("comp1").func("an3").set("args", "x, y,z");
    model.component("comp1").func("an3").setIndex("argunit", "m", 2);
    model.component("comp1").func("an3").set("args", "x, y");

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(((Yee/2)*(emw.Eupx+emw.Edownx))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z)))))*cos(T(x,y,z))*cos(P(x,y))", "(((Yee/2)*(emw.Eupy+emw.Edowny))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,z)-d(emw.Eupy,x))+(d(emw.Edownz,z)-d(emw.Edowny,x)))))*cos(T(x,y,z))*sin(P(x,y))", "(((Yee/2)*(emw.Eupz+emw.Edownz))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,x)-d(emw.Eupy,y))+(d(emw.Edownz,x)-d(emw.Edowny,y)))))*(-sin(T(x,y,z)))"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"(((Zmm/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z))))-((Xem/2)*(emw.Eupx+emw.Edownx)))*(-sin(P(x,y)))", "(((Zmm/2)*(-1/(j*w*mu0))*((d(emw.Eupz,z)-d(emw.Eupy,x))+(d(emw.Edownz,z)-d(emw.Edowny,x))))-((Xem/2)*(emw.Eupy+emw.Edowny)))*(cos(P(x,y)))", "0"});

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"(((Yee/2)*(emw.Eupx+emw.Edownx))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z)))))*cos(T(x,y,z))*cos(P(x,y))", "(((Yee/2)*(emw.Eupy+emw.Edowny))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupx,z)-d(emw.Eupz,x))+(d(emw.Edownx,z)-d(emw.Edownz,x)))))*cos(T(x,y,z))*sin(P(x,y))", "(((Yee/2)*(emw.Eupz+emw.Edownz))+((Xem/2)*(-1/(j*w*mu0))*((d(emw.Eupy,x)-d(emw.Eupx,y))+(d(emw.Edowny,x)-d(emw.Edownx,y)))))*(-sin(T(x,y,z)))"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"(((Zmm/2)*(-1/(j*w*mu0))*((d(emw.Eupz,y)-d(emw.Eupy,z))+(d(emw.Edownz,y)-d(emw.Edowny,z))))-((Xem/2)*(emw.Eupx+emw.Edownx)))*(-sin(P(x,y)))", "(((Zmm/2)*(-1/(j*w*mu0))*((d(emw.Eupx,z)-d(emw.Eupz,x))+(d(emw.Edownx,z)-d(emw.Edownz,x))))-((Xem/2)*(emw.Eupy+emw.Edowny)))*(cos(P(x,y)))", "0"});

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg2").run();
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "emw.normJs");
    model.result("pg5").feature("surf1").set("descr", "Surface current density norm");
    model.result("pg5").run();

    model.param().set("Yee", "-j*0.0036679[S]");
    model.param().set("Zmm", "-j*184.0435[ohm]");

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("adaptmesh1").run("se1");

    model.study("std1").feature("fdrfam").set("saveres", false);
    model.study("std1").feature("fdrfam").set("goalorientedtermination", "off");
    model.study("std1").feature("fdrfam").set("errestim", "goalerrest");
    model.study("std1").feature("fdrfam").set("goalfunctype", "gfman");
    model.study("std1").feature("fdrfam").set("gfunc", "comp1.emw.intWe");

    model.sol("sol1").study("std1");

    model.study("std1").feature("fdrfam").set("notlistsolnum", 1);
    model.study("std1").feature("fdrfam").set("notsolnum", "auto");
    model.study("std1").feature("fdrfam").set("listsolnum", 1);
    model.study("std1").feature("fdrfam").set("solnum", "auto");

    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "fdrfam");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "fdrfam");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").feature("eeDef").set("adjppr", "on");
    model.sol("sol1").feature("s1").feature("adDef").set("selection", "sum");
    model.sol("sol1").feature("s1").feature("adDef").set("globalminpar", 1.4);
    model.sol("sol1").feature("s1").feature("adDef").set("classicadap", true);
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "fdrfam");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.study("std1").feature("fdrfam").active(false);
    model.study("std1").feature("freq").active(true);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");

    return model;
  }

  public static Model run4(Model model) {
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg3").run();
    model.result("pg5").run();

    model.component("comp1").view("view1").set("transparency", false);

    model.component("comp1").physics("emw").feature("scu1").active(false);
    model.component("comp1").physics("emw").feature("smcd1").active(false);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg3").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").set("data", "dset1");
    model.result("pg1").run();
    model.result().dataset().create("cln1", "CutLine3D");
    model.result().dataset("cln1").set("method", "pointdir");
    model.result().dataset("cln1").set("pddir", new int[]{1, 0, 1});
    model.result().dataset("cln1").set("method", "twopoint");
    model.result().dataset("cln1").setIndex("genpoints", 1, 1, 2);
    model.result().create("pg6", "PlotGroup1D");
    model.result("pg6").run();
    model.result("pg6").create("lngr1", "LineGraph");
    model.result("pg6").feature("lngr1").set("markerpos", "datapoints");
    model.result("pg6").feature("lngr1").set("linewidth", "preference");
    model.result("pg6").run();
    model.result("pg6").set("data", "cln1");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("expr", "real(emw.Ex)");
    model.result("pg6").run();

    model.component("comp1").mesh("mesh1").automatic(false);

    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/(10*sqrt(er))");
    model.component("comp1").mesh("mesh1").feature("size1").set("hmin", "lam/100000");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();

    model.component("comp1").physics("emw").feature("scu1").active(true);
    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"Yee*emw.Ex-Xem*(sin(T(x,y,z))*sin(P(x,y))*emw.Hz-cos(T(x,y,z))*emw.Hy)", "Yee*emw.Ey-Xem*(cos(T(x,y,z))*emw.Hx-sin(T(x,y,z))*cos(P(x,y))*emw.Hz)", "Yee*emw.Ez-Xem*(sin(T(x,y,z))*cos(P(x,y))*emw.Hy-sin(T(x,y,z))*sin(P(x,y))*emw.Hx)"});
    model.component("comp1").physics("emw").feature("smcd1").active(true);
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx-Xem*(sin(T(x,y,z))*sin(P(x,y))*emw.Ez-cos(T(x,y,z))*emw.Ey)", "Zmm*emw.Hy -Xem*(cos(T(x,y,z))*emw.Ex-sin(T(x,y,z))*cos(P(x,y))*emw.Ez)", "Zmm*emw.Hz-Xem*(sin(T(x,y,z))*cos(P(x,y))*emw.Ey-sin(T(x,y,z))*sin(P(x,y))*emw.Ex)"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg4").run();
    model.result("pg1").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"Yee*emw.Ex", "Yee*emw.Ey", "Yee*emw.Ez"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx", "Zmm*emw.Hy", "Zmm*emw.Hz"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();

    model.component("comp1").geom("geom1").run("pt1");
    model.component("comp1").geom("geom1").create("ls1", "LineSegment");
    model.component("comp1").geom("geom1").feature("ls1").selection("vertex1").named("pt1");
    model.component("comp1").geom("geom1").feature("ls1").selection("vertex1").set("pt1", 1);
    model.component("comp1").geom("geom1").feature("ls1").set("specify1", "coord");
    model.component("comp1").geom("geom1").feature("ls1").set("specify2", "coord");
    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new int[]{1, 0, 1});
    model.component("comp1").geom("geom1").runPre("fin");

    model.param().set("blkDim", "2*a+4*pml");

    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "blkDim", 0);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "blkDim", 1);
    model.component("comp1").geom("geom1").feature("blk1").setIndex("size", "blkDim", 2);
    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"blkDim*sqrt(2)", "0", "1"});
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)", 2);
    model.component("comp1").geom("geom1").run("ls1");
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)/4", 0);
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)/4", 2);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)/2", 2);
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)/2", 0);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)", 0);
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim*sqrt(2)", 2);
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature("ls1").setIndex("coord2", "blkDim/2", 0);
    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"blkDim/2", "0", "blkDim*2"});
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"blkDim/2", "0", "blkDim/2"});
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").run();

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").attach("std1");

    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"a", "0", "a"});
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"a+pml", "0", "a+pml"});
    model.component("comp1").geom("geom1").runPre("fin");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").set("data", "dset1");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").selection().set(90, 95);
    model.result("pg6").run();

    model.param().set("b", "2*lam");
    model.param().set("blkDim", "2*b+4*pml");

    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"b+pml", "0", "b+pml"});
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").runPre("fin");
    model.component("comp1").geom("geom1").runPre("fin");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("data", "cln1");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("resolution", "extrafine");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("resolution", "normal");
    model.result("pg6").feature("lngr1").set("smooth", "everywhere");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("data", "dset1");
    model.result("pg6").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "nxc");
    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"Yee*emw.Ex-Xem*(nyc*emw.Hz-nzc*emw.Hy)", "Yee*emw.Ey", "Yee*emw.Ez"});

    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "nyc");
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "nzc");
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "nz");
    model.result("pg5").run();

    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"Yee*emw.Ex-Xem*(ny*emw.Hz-nz*emw.Hy)", "Yee*emw.Ey-Xem*(nx*emw.Hz-nz*emw.Hx)", "Yee*emw.Ez-Xem*(nx*emw.Hy-ny*emw.Hx)"});
    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx-Xem*(ny*emw.Ez-nz*emw.Ey)", "Zmm*emw.Hy-Xem*(nx*emw.Ez-nz*emw.Ex)", "Zmm*emw.Hz-Xem*(nx*emw.Ey-ny*emw.Ex)"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");

    return model;
  }

  public static Model run5(Model model) {
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx-Xem*(ny*emw.Ez-nz*emw.Ey)", "Zmm*emw.Hy+Xem*(nx*emw.Ez-nz*emw.Ex)", "Zmm*emw.Hz-Xem*(nx*emw.Ey-ny*emw.Ex)"});
    model.component("comp1").physics("emw").feature("scu1")
         .set("Js0", new String[]{"Yee*emw.Ex-Xem*(ny*emw.Hz-nz*emw.Hy)", "Yee*emw.Ey+Xem*(nx*emw.Hz-nz*emw.Hx)", "Yee*emw.Ez-Xem*(nx*emw.Hy-ny*emw.Hx)"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg1").run();
    model.result().create("pg7", "PlotGroup3D");
    model.result("pg7").run();
    model.result("pg7").create("arws1", "ArrowSurface");
    model.result("pg7").feature("arws1").setIndex("expr", "nx", 0);
    model.result("pg7").feature("arws1").set("expr", new String[]{"nx", "ny", "nz"});
    model.result("pg7").run();

    model.component("comp1").view("view1").set("transparency", false);

    model.result("pg7").run();
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(73);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(69, 73);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(69, 70, 73);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(69, 70, 73, 74);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(69, 70, 72, 73, 74);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(68, 69, 70, 72, 73, 74);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(67, 68, 69, 70, 72, 73, 74);
    model.result("pg7").selection().geom("geom1", 2);
    model.result("pg7").selection().set(67, 68, 69, 70, 71, 72, 73, 74);
    model.result("pg7").run();
    model.result("pg6").run();
    model.result("pg1").run();

    model.label("DipoleInSphere.mph");

    model.component("comp1").mesh("mesh1").automatic(true);
    model.component("comp1").mesh("mesh1").autoMeshSize(5);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();

    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx+Xem*(ny*emw.Ez-nz*emw.Ey)", "Zmm*emw.Hy-Xem*(nx*emw.Ez-nz*emw.Ex)", "Zmm*emw.Hz+Xem*(nx*emw.Ey-ny*emw.Ex)"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();

    model.component("comp1").physics("emw").feature("smcd1")
         .set("Jms0", new String[]{"Zmm*emw.Hx-Xem*(ny*emw.Ez-nz*emw.Ey)", "Zmm*emw.Hy+Xem*(nx*emw.Ez-nz*emw.Ex)", "Zmm*emw.Hz-Xem*(nx*emw.Ey-ny*emw.Ex)"});

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").create("size2", "Size");
    model.component("comp1").mesh("mesh1").feature().move("size2", 3);
    model.component("comp1").mesh("mesh1").feature("size2").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("size2").selection().geom("geom1", 2);
    model.component("comp1").mesh("mesh1").feature("size2").selection().set(67, 68, 69, 70, 71, 72, 73, 74);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/60");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result().export().create("plot1", "pg6", "lngr1", "Plot");
    model.result().export("plot1")
         .set("filename", "E:\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\COMSOL Simulation\\COMSOL.txt");
    model.result().export("plot1").set("header", false);
    model.result().export("plot1").run();

    model.component("comp1").geom("geom1").feature("ls1").set("coord2", new String[]{"b", "0", "b"});

    model.param().set("blkDim", "2*b+2*pml");

    model.component("comp1").geom("geom1").run();

    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/80");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result().export("plot1").run();
    model.result().export("plot1")
         .set("filename", "E:\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\SIE-GSTC Code Biisotropy\\COMSOL.txt");
    model.result().export("plot1").run();

    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/100");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result().export("plot1")
         .set("filename", "E:\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\SIE-GSTC Code Biisotropy\\COMSOL_MoreMesh.txt");
    model.result().export("plot1").run();

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/60");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").attach("std1");

    model.result("pg6").run();
    model.result("pg1").run();

    model.label("DipoleInSphere.mph");

    return model;
  }

  public static Model run6(Model model) {

    model.result("pg1").run();

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/30");

    model.sol("sol1").study("std1");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("mesh1").automatic(true);

    model.component("comp1").physics("emw").feature("scu1").active(false);
    model.component("comp1").physics("emw").feature("smcd1").active(false);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result().export("plot1")
         .set("filename", "E:\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\SIE-GSTC Code Biisotropy\\COMSOL_NoMS.txt");
    model.result().export("plot1").run();

    model.component("comp1").physics("emw").feature("scu1").active(true);
    model.component("comp1").physics("emw").feature("smcd1").active(true);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("smooth", "internal");
    model.result("pg6").feature("lngr1").set("recover", "pprint");
    model.result("pg6").feature("lngr1").set("threshold", "manual");
    model.result("pg6").feature("lngr1").set("resolution", "extrafine");
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("thresholdvalue", 0.01);
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("thresholdvalue", 1);
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("threshold", "none");
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").set("data", "cln1");
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("data", "cln1");
    model.result("pg6").run();
    model.result().dataset("cln1").setIndex("genpoints", "b", 1, 0);
    model.result().dataset("cln1").setIndex("genpoints", "b", 1, 2);
    model.result("pg6").run();

    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/60");

    model.component("comp1").physics("emw").feature("scu1").active(false);
    model.component("comp1").physics("emw").feature("smcd1").active(false);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/30");
    model.component("comp1").mesh("mesh1").create("dis2", "Distribution");
    model.component("comp1").mesh("mesh1").feature().move("dis2", 2);
    model.component("comp1").mesh("mesh1").feature("dis2").selection().geom("geom1", 1);
    model.component("comp1").mesh("mesh1").feature("dis2").selection().set(95);

    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").mesh("mesh1").feature("dis2").selection().set(90, 95);

    model.component("comp1").view("view1").set("transparency", false);

    model.component("comp1").mesh("mesh1").feature("dis2").set("numelem", 1000);
    model.component("comp1").mesh("mesh1").run();

    model.component("comp1").view("view1").set("transparency", true);

    model.component("comp1").mesh("mesh1").feature("dis2").set("numelem", 5000);
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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result().export("plot1").run();

    model.component("comp1").physics("emw").feature("scu1").active(true);
    model.component("comp1").physics("emw").feature("smcd1").active(true);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();

    model.component("comp1").view("view1").set("hidestatus", "hide");

    model.component("comp1").mesh("mesh1").create("size2", "Size");
    model.component("comp1").mesh("mesh1").feature().move("size2", 2);
    model.component("comp1").mesh("mesh1").feature("size2").selection().geom("geom1", 2);
    model.component("comp1").mesh("mesh1").feature("size2").selection().set(67, 68, 69, 70, 71, 72, 73, 74);
    model.component("comp1").mesh("mesh1").feature("size2").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/60");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result().export("plot1")
         .set("filename", "E:\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\SIE-GSTC Code Biisotropy\\COMSOL.txt");
    model.result().export("plot1").run();
    model.result("pg6").run();
    model.result("pg6").feature("lngr1").set("data", "dset1");
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg6").run();
    model.result("pg5").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "emw.normJs");
    model.result("pg5").feature("surf1").set("descr", "Surface current density norm");
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "emw.smcd1.Jmsx");
    model.result("pg5").feature("surf1").set("descr", "Surface magnetic current density, x-component");
    model.result("pg5").feature("surf1")
         .set("expr", "sqrt(abs(emw.smcd1.Jmsx)^2+abs(emw.smcd1.Jmsy)^2+abs(emw.smcd1.Jmsz)^2)");
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("rangecoloractive", true);
    model.result("pg5").feature("surf1").set("rangecolormax", 1000000);
    model.result("pg5").feature("surf1").set("rangecolormin", 188353.060006932);
    model.result("pg5").feature("surf1").set("rangecolormax", 1500000);
    model.result("pg6").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "arg(emw.smcd1.Jmsy)");
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("rangecoloractive", false);
    model.result("pg6").run();
    model.result("pg5").run();

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/50");

    model.sol("sol1").study("std1");
    model.sol("sol1").feature().remove("st1");
    model.sol("sol1").feature().remove("v1");
    model.sol("sol1").feature().remove("s1");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/40");
    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/100");
    model.component("comp1").mesh("mesh1").feature("size1").set("hmax", "lam/30");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.component("comp1").mesh("mesh1").feature().clear();
    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").automatic(true);
    model.component("comp1").mesh("mesh1").run();
    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").create("size2", "Size");
    model.component("comp1").mesh("mesh1").feature().move("size2", 2);
    model.component("comp1").mesh("mesh1").feature("size2").selection().geom("geom1", 2);
    model.component("comp1").mesh("mesh1").feature("size2").selection().set(67, 68, 69, 70, 71, 72, 73, 74);
    model.component("comp1").mesh("mesh1").feature("size2").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/100");
    model.component("comp1").mesh("mesh1").create("dis2", "Distribution");

    model.component("comp1").view("view1").set("renderwireframe", true);

    model.component("comp1").mesh("mesh1").feature("dis2").selection().geom("geom1", 1);
    model.component("comp1").mesh("mesh1").feature("dis2").selection().set(90, 95);
    model.component("comp1").mesh("mesh1").feature("dis2").set("numelem", 5000);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");

    return model;
  }

  public static Model run7(Model model) {
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    model.label("DipoleInSphere.mph");

    model.component("comp1").mesh("mesh1").feature().clear();
    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").automatic(true);
    model.component("comp1").mesh("mesh1").automatic(false);
    model.component("comp1").mesh("mesh1").create("dis2", "Distribution");
    model.component("comp1").mesh("mesh1").feature("dis2").selection().geom("geom1", 1);
    model.component("comp1").mesh("mesh1").feature("dis2").selection().set(90, 95);
    model.component("comp1").mesh("mesh1").feature().move("dis2", 3);
    model.component("comp1").mesh("mesh1").feature("dis2").set("numelem", 1000);

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg6").run();

    model.component("comp1").mesh("mesh1").create("size2", "Size");
    model.component("comp1").mesh("mesh1").feature().move("size2", 2);
    model.component("comp1").mesh("mesh1").feature("size2").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/50");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");

    model.label("DipoleInSphere.mph");

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");

    model.component("comp1").mesh("mesh1").feature("size2").selection().geom("geom1", 2);
    model.mesh().clearMeshes();
    model.component("comp1").mesh("mesh1").feature("size2").selection().set(73, 74);

    model.component("comp1").view("view1").set("transparency", false);

    model.component("comp1").mesh("mesh1").feature("size2").selection().set(67, 68, 69, 70, 71, 72, 73, 74);

    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg6").run();
    model.result("pg7").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "abs(emw.smcd1.Jmsy)");
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg6").run();

    model.component("comp1").mesh("mesh1").feature("size2").set("hmax", "lam/100");

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
    model.sol("sol1").feature("s1").set("stol", 0.01);
    model.sol("sol1").feature("s1").create("p1", "Parametric");
    model.sol("sol1").feature("s1").feature().remove("pDef");
    model.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{"freq"});
    model.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{"f"});
    model.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{"GHz"});
    model.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "no");
    model.sol("sol1").feature("s1").feature("p1").set("preusesol", "auto");
    model.sol("sol1").feature("s1").feature("p1").set("pdistrib", "off");
    model.sol("sol1").feature("s1").feature("p1").set("plot", "on");
    model.sol("sol1").feature("s1").feature("p1").set("plotgroup", "pg1");
    model.sol("sol1").feature("s1").feature("p1").set("probesel", "all");
    model.sol("sol1").feature("s1").feature("p1").set("probes", new String[]{});
    model.sol("sol1").feature("s1").feature("p1").set("control", "freq");
    model.sol("sol1").feature("s1").set("linpmethod", "sol");
    model.sol("sol1").feature("s1").set("linpsol", "zero");
    model.sol("sol1").feature("s1").set("control", "freq");
    model.sol("sol1").feature("s1").feature("aDef").set("complexfun", true);
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").label("Suggested Direct Solver (emw)");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "tfqmr");
    model.sol("sol1").feature("s1").feature("i1").set("prefuntype", "right");
    model.sol("sol1").feature("s1").feature("i1").label("Suggested Iterative Solver (emw)");
    model.sol("sol1").feature("s1").feature("i1").create("sv1", "SORVector");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("iter", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("relax", "1");
    model.sol("sol1").feature("s1").feature("i1").feature("sv1").set("seconditer", "3");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");
    model.sol("sol1").runAll();

    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg7").run();
    model.result("pg6").run();
    model.result().export("plot1")
         .set("filename", "C:\\Users\\JordanBudhu\\Dropbox\\Research Projects VT\\Conferences\\APS 2026\\Budhu_Dault\\SIE-GSTC Code Biisotropy\\COMSOL.txt");
    model.result().export("plot1").run();
    model.result().export("plot1").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("expr", "arg(emw.smcd1.Jmsy)");
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg5").feature("surf1")
         .set("expr", "sqrt(abs(emw.smcd1.Jmsx)^2+abs(emw.smcd1.Jmsy)^2+abs(emw.smcd1.Jmsz)^2)");
    model.result("pg5").run();
    model.result("pg5").feature("surf1").set("rangecoloractive", true);
    model.result("pg5").feature("surf1").set("rangecolormin", 0);
    model.result("pg5").feature("surf1").set("rangecolormax", 800000);
    model.result("pg5").run();
    model.result("pg1").run();
    model.result("pg5").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg4").feature("rp2").active(false);
    model.result("pg4").run();
    model.result("pg4").run();
    model.result("pg1").run();

    model.label("DipoleInSphereMoreMesh.mph");

    model.result("pg1").run();

    return model;
  }

  public static void main(String[] args) {
    Model model = run();
    model = run2(model);
    model = run3(model);
    model = run4(model);
    model = run5(model);
    model = run6(model);
    run7(model);
  }

}
