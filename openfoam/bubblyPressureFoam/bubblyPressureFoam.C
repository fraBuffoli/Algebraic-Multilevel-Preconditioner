/*---------------------------------------------------------------------------*\
Application
    bubblyPressureFoam

Description
    Reference solves with the OpenFOAM linear solvers (GAMG, PCG, ...) on the pressure matrices of
    the bubbles3d generator of schwarz2lvl, for a comparison of the two-level Schwarz preconditioner
    with OpenFOAM on exactly the same systems.

    Pressure equation of an incompressible two-phase (VoF) flow on the unit cube:
        -div(beta grad p) = alpha,   beta = 1/rho,   rho = alpha rho_g + (1 - alpha) rho_l,
    rho_l = 1, rho_g = 1/rhoRatio, alpha = gas fraction of rising bubbles. alpha comes from
    include/utils/bubble_field.hpp of schwarz2lvl, compiled with the schwarz2lvl compiler and linked
    through a C interface (bubbleFieldCapi.h): same bubbles, same machine code as the generator.
    It is evaluated at the cell centres, so any mesh decomposition gives the same system. With
        laplacianSchemes { default Gauss linear orthogonal; }
        interpolationSchemes { default linear; }
    on the uniform blockMesh of the case, p fixedValue 0 on "top" and zeroGradient elsewhere, the
    matrix and right-hand side are those of `schwarz2lvl --generate bubbles3d --rhs generated` up to
    round-off (checked with -writeMatrix against `matgen bubbles3d ... t=<last step>`).

    For every step (pseudo-time = step, bubbles rising by cfl cells per step) and every corrector the
    equation is assembled and solved from p = 0 (as schwarz2lvl does) with the solver controls
    solvers/<solver> of system/fvSolution. Every solve is timed between two synchronizations of all
    processes (as schwarz2lvl does); assembly is not timed. The true relative residual
    ||b - A x||_2 / ||b||_2 is computed after every solve: the stopping test of OpenFOAM is on the
    normalised L1 residual (with p = 0 at the start: sum|b - A x| / sum|b|).

Usage
    bubblyPressureFoam [-parallel] [-solver NAME] [-steps N] [-correctors N] [-csv FILE]
                       [-writeMatrix PREFIX] [-warmStart]

    Parameters of the bubbles: system/bubblesDict (nx ny nz nBubbles radius rhoRatio cfl seed steps
    correctors). One CSV line per solve, written by the master process.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "clockTime.H"
#include "OFstream.H"
#include "zeroGradientFvPatchFields.H"

#include "bubbleFieldCapi.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace Foam;

namespace
{

// All processes reach this point before any leaves it.
void synchronize()
{
    if (Pstream::parRun())
    {
        returnReduce(label(1), sumOp<label>());
    }
}

// Right-hand side of the system the linear solvers see (fvScalarMatrix::solveSegregated): source
// with the contributions of the uncoupled patches (the coupled ones go through the interfaces).
tmp<scalarField> totalSource(fvScalarMatrix& eqn)
{
    const volScalarField& psi = eqn.psi();
    const fvMesh& mesh = psi.mesh();
    tmp<scalarField> tb(new scalarField(eqn.source()));
    scalarField& b = tb.ref();
    forAll(psi.boundaryField(), patchi)
    {
        if (!psi.boundaryField()[patchi].coupled())
        {
            const labelUList& fc = mesh.boundary()[patchi].faceCells();
            const scalarField& bc = eqn.boundaryCoeffs()[patchi];
            forAll(fc, facei)
            {
                b[fc[facei]] += bc[facei];
            }
        }
    }
    return tb;
}

// True residual b - A x of the system the linear solvers see: diagonal with the boundary
// contributions, coupled (processor) patches through the interfaces. fvMatrix::residual() adds the
// coupled contributions twice in parallel, so it is not used. Returns ||b - A x||_2 / ||b||_2.
scalar trueRelativeResidual(fvScalarMatrix& eqn)
{
    const volScalarField& psi = eqn.psi();
    const scalarField b(totalSource(eqn));
    const scalarField saveDiag(eqn.diag());
    eqn.diag() = eqn.D();
    scalarField r(psi.size());
    eqn.lduMatrix::residual
    (
        r,
        psi.primitiveField(),
        b,
        eqn.boundaryCoeffs(),
        psi.boundaryField().scalarInterfaces(),
        0
    );
    eqn.diag() = saveDiag;
    return Foam::sqrt(gSumSqr(r))/max(Foam::sqrt(gSumSqr(b)), vSmall);
}


// Writes the matrix (global cell numbers i + nx (j + ny k), 1-based) and the right-hand side in
// Matrix Market format (serial runs only).
void writeMatrixMarket
(
    const fileName& prefix,
    const fvScalarMatrix& eqn,
    const labelList& gid
)
{
    const fvMesh& mesh = eqn.psi().mesh();
    const labelUList& l = mesh.lduAddr().lowerAddr();
    const labelUList& u = mesh.lduAddr().upperAddr();
    const scalarField diag(eqn.D());   // with the boundary contributions
    const scalarField& upper = eqn.upper();
    const scalarField& lower = eqn.lower();
    const label n = mesh.nCells();

    std::FILE* f = std::fopen((prefix + ".mtx").c_str(), "w");
    if (!f)
    {
        FatalErrorInFunction << "cannot write " << prefix << ".mtx" << exit(FatalError);
    }
    std::fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    std::fprintf(f, "%d %d %d\n", int(n), int(n), int(n + 2*l.size()));
    for (label c = 0; c < n; ++c)
    {
        std::fprintf(f, "%d %d %.17g\n", int(gid[c] + 1), int(gid[c] + 1), diag[c]);
    }
    forAll(l, facei)
    {
        std::fprintf(f, "%d %d %.17g\n", int(gid[l[facei]] + 1), int(gid[u[facei]] + 1), upper[facei]);
        std::fprintf(f, "%d %d %.17g\n", int(gid[u[facei]] + 1), int(gid[l[facei]] + 1), lower[facei]);
    }
    std::fclose(f);

    scalarField b(eqn.source());
    std::vector<double> bg(n, 0.0);
    forAll(b, c)
    {
        bg[gid[c]] = b[c];
    }
    f = std::fopen((prefix + "_rhs.mtx").c_str(), "w");
    std::fprintf(f, "%%%%MatrixMarket matrix array real general\n%d 1\n", int(n));
    for (label c = 0; c < n; ++c)
    {
        std::fprintf(f, "%.17g\n", bg[c]);
    }
    std::fclose(f);
}

} // namespace


int main(int argc, char *argv[])
{
    argList::addOption("solver", "name", "solver controls solvers/<name> in system/fvSolution (default p)");
    argList::addOption("steps", "N", "number of steps (default: steps in system/bubblesDict)");
    argList::addOption("correctors", "N", "solves per step (default: correctors in system/bubblesDict)");
    argList::addOption("csv", "file", "CSV with one line per solve (default bubbly_<solver>.csv)");
    argList::addOption("writeMatrix", "prefix", "write matrix and rhs of the last step (serial only)");
    argList::addBoolOption("warmStart", "start every solve from the previous solution instead of 0");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    IOdictionary bubblesDict
    (
        IOobject("bubblesDict", runTime.system(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE)
    );

    const word solverName(args.optionLookupOrDefault<word>("solver", "p"));
    const label nx = bubblesDict.lookup<label>("nx");
    const label ny = bubblesDict.lookupOrDefault<label>("ny", nx);
    const label nz = bubblesDict.lookupOrDefault<label>("nz", nx);
    const scalar rhoRatio = bubblesDict.lookup<scalar>("rhoRatio");
    const label steps = args.optionLookupOrDefault<label>("steps", bubblesDict.lookupOrDefault<label>("steps", 1));
    const label correctors =
        args.optionLookupOrDefault<label>("correctors", bubblesDict.lookupOrDefault<label>("correctors", 1));
    const bool warmStart = args.optionFound("warmStart");
    const fileName csvName(args.optionLookupOrDefault<fileName>("csv", fileName("bubbly_" + solverName + ".csv")));

    const label nBubbles = bubblesDict.lookup<label>("nBubbles");
    const scalar radius = bubblesDict.lookup<scalar>("radius");
    const scalar cfl = bubblesDict.lookup<scalar>("cfl");
    const label seed = bubblesDict.lookupOrDefault<label>("seed", 2024);
    // Bubble field of schwarz2lvl at pseudo-time t (C interface, see bubbleFieldCapi.h).
    auto newField = [&](const scalar t)
    {
        void* f = s2lBubbleFieldNew(3, nx, ny, nz, nBubbles, radius, cfl, t, unsigned(seed));
        if (!f)
        {
            FatalErrorInFunction << "invalid bubble parameters in bubblesDict" << exit(FatalError);
        }
        return f;
    };

    const label nGlobal = returnReduce(mesh.nCells(), sumOp<label>());
    if (nGlobal != nx*ny*nz)
    {
        FatalErrorInFunction
            << "the mesh has " << nGlobal << " cells, bubblesDict gives " << nx << "x" << ny << "x" << nz
            << exit(FatalError);
    }
    if (rhoRatio <= 0)
    {
        FatalErrorInFunction << "rhoRatio must be > 0" << exit(FatalError);
    }
    const dictionary& controls = mesh.solution().solverDict(solverName);

    // Cell (i, j, k) of every local cell from its centre: global number and bubble lookups.
    std::vector<int> ci(mesh.nCells()), cj(mesh.nCells()), ck(mesh.nCells());
    labelList gid(mesh.nCells());
    {
        void* grid = newField(0);
        const volVectorField& C = mesh.C();
        forAll(C, c)
        {
            s2lBubbleFieldCellOf(grid, C[c].x(), C[c].y(), C[c].z(), &ci[c], &cj[c], &ck[c]);
            gid[c] = ci[c] + nx*(cj[c] + ny*ck[c]);
        }
        s2lBubbleFieldDelete(grid);
    }
    std::vector<double> alphaCells(mesh.nCells());

    volScalarField p
    (
        IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField alpha
    (
        IOobject("alpha", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimless, 0),
        zeroGradientFvPatchScalarField::typeName
    );
    const dimensionedScalar rhoG(dimless, 1.0/rhoRatio);

    autoPtr<OFstream> csv;
    if (Pstream::master())
    {
        csv.reset(new OFstream(csvName));
        csv() << "solver,np,ncells,step,corrector,t_solve,iterations,initial_res,final_res,true_rel_res,converged,"
                 "t_setup,t_cycles,t_assembly" << nl;
    }

    Info<< "bubblyPressureFoam: " << nx << "x" << ny << "x" << nz << " cells, " << nBubbles
        << " bubbles, radius " << radius << ", rho ratio " << rhoRatio << ", cfl " << cfl << ", seed " << seed << ", "
        << steps << " steps x " << correctors << " correctors, solver " << solverName << " "
        << controls << nl << endl;

    scalarField times;
    labelList iters;
    for (label step = 0; step < steps; ++step)
    {
        // New gas fraction and face coefficients (timed apart, not part of the solver cost).
        synchronize();
        const clockTime assemblyTimer;
        void* bubbles = newField(step);
        s2lBubbleFieldAlpha(bubbles, mesh.nCells(), ci.data(), cj.data(), ck.data(), alphaCells.data());
        s2lBubbleFieldDelete(bubbles);
        scalarField& a = alpha.primitiveFieldRef();
        forAll(a, c)
        {
            a[c] = alphaCells[c];
        }
        alpha.correctBoundaryConditions();
        Info<< "step " << step << ": sum of alpha " << gSum(alpha.primitiveField()) << endl;
        const volScalarField beta(1.0/(alpha*rhoG + (1.0 - alpha)));
        const surfaceScalarField betaf(fvc::interpolate(beta));
        scalar tStepAssembly = returnReduce(assemblyTimer.elapsedTime(), maxOp<scalar>());

        for (label corr = 0; corr < correctors; ++corr)
        {
            synchronize();
            const clockTime eqnTimer;
            fvScalarMatrix pEqn(-fvm::laplacian(betaf, p));
            pEqn.source() += mesh.V().field()*alpha.primitiveField();
            const scalar tAssembly = returnReduce(eqnTimer.elapsedTime(), maxOp<scalar>()) + tStepAssembly;
            tStepAssembly = 0;

            if (!warmStart)
            {
                p.primitiveFieldRef() = 0;
                p.correctBoundaryConditions();
            }
            if (step == steps - 1 && corr == 0 && args.optionFound("writeMatrix"))
            {
                if (Pstream::parRun())
                {
                    FatalErrorInFunction << "-writeMatrix: serial runs only" << exit(FatalError);
                }
                writeMatrixMarket(args.optionRead<fileName>("writeMatrix"), pEqn, gid);
            }

            // As fvMatrix<scalar>::solveSegregated (pEqn.solve), with the construction of the solver
            // (GAMG: agglomeration of the coarse matrices; the agglomeration itself is cached) and
            // its iterations timed separately, each between two synchronizations of all processes.
            const scalarField saveDiag(pEqn.diag());
            pEqn.diag() = pEqn.D();
            const scalarField source(totalSource(pEqn));
            synchronize();
            const clockTime setupTimer;
            autoPtr<lduMatrix::solver> solver = lduMatrix::solver::New
            (
                p.name(),
                pEqn,
                pEqn.boundaryCoeffs(),
                pEqn.internalCoeffs(),
                p.boundaryField().scalarInterfaces(),
                controls
            );
            const scalar tSetup = returnReduce(setupTimer.elapsedTime(), maxOp<scalar>());
            const clockTime cyclesTimer;
            const solverPerformance perf = solver->solve(p.primitiveFieldRef(), source);
            const scalar tCycles = returnReduce(cyclesTimer.elapsedTime(), maxOp<scalar>());
            pEqn.diag() = saveDiag;
            p.correctBoundaryConditions();
            perf.print(Info);
            const scalar t = tSetup + tCycles;

            const scalar rel = trueRelativeResidual(pEqn);
            times.append(t);
            iters.append(perf.nIterations());

            Info<< "step " << step << " corrector " << corr << ": " << perf.nIterations() << " iterations, "
                << t << " s, true relative residual " << rel << ", ||p||_2 "
                << Foam::sqrt(gSumSqr(p.primitiveField())) << endl;
            if (Pstream::master())
            {
                csv() << solverName << ',' << Pstream::nProcs() << ',' << nGlobal << ',' << step << ','
                      << corr << ',' << t << ',' << perf.nIterations() << ',' << perf.initialResidual()
                      << ',' << perf.finalResidual() << ',' << rel << ',' << (perf.converged() ? 1 : 0)
                      << ',' << tSetup << ',' << tCycles << ',' << tAssembly << endl;
            }
        }
    }

    if (times.size())
    {
        scalarField sorted(times);
        std::sort(sorted.begin(), sorted.end());
        Info<< nl << "Summary (" << solverName << "): " << times.size() << " solves, total " << sum(times)
            << " s, median " << sorted[sorted.size()/2] << " s per solve, iterations "
            << sum(iters)/scalar(iters.size()) << " on average, first solve " << times[0] << " s" << nl
            << endl;
    }

    Info<< "End" << nl << endl;
    return 0;
}


// ************************************************************************* //
